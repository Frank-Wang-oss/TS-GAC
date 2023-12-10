import torch
import numpy as np

class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)

        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    def _get_correlated_mask_node(self, num_node, batch):
        diag = np.eye(2 * num_node)
        l1 = np.eye((2 * num_node), 2 * num_node, k=-num_node)
        l2 = np.eye((2 * num_node), 2 * num_node, k=num_node)

        diag = np.repeat(np.expand_dims(diag, 0), batch, axis=0)
        l1 = np.repeat(np.expand_dims(l1, 0), batch, axis=0)
        l2 = np.repeat(np.expand_dims(l2, 0), batch, axis=0)


        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    def _get_correlated_mask_node_cross_view(self, num_node, batch):
        diag = np.eye(num_node)

        diag = np.repeat(np.expand_dims(diag, 0), batch, axis=0)


        mask = torch.from_numpy(diag)
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y, dim1, dim2):
        v = torch.tensordot(x.unsqueeze(dim1), y.T.unsqueeze(dim2), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y, dim1, dim2):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(dim1), y.unsqueeze(dim2))
        return v

    def forward(self, zis, zjs):
        batch, num_nodes, feature_dimension = zis.size()
        zis_nodes = torch.clone(zis)
        zjs_nodes = torch.clone(zjs)
        # zis_nodes = zis.copy()
        # zjs_nodes = zjs.copy()

        zis = torch.reshape(zis, [batch, -1])
        zjs = torch.reshape(zjs, [batch, -1])

        ### graph-level features contextual contrasting
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations, 1, 0)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size) ### size (batch)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)


        ### node-level features contextual contrasting

        similarity_matrix_nodes_1to2 = self.similarity_function(zis_nodes, zjs_nodes, 1, 2)
        similarity_matrix_nodes_2to1 = self.similarity_function(zjs_nodes, zis_nodes, 1, 2)

        l_pos_node = torch.diagonal(similarity_matrix_nodes_1to2, offset=0, dim1=-1, dim2=-2) ### size (batch, num_nodes)
        r_pos_node = torch.diagonal(similarity_matrix_nodes_2to1, offset=0, dim1=-1, dim2=-2)

        mask_samples_from_same_repr_node = self._get_correlated_mask_node_cross_view(num_nodes, batch).type(torch.bool)
        l_positives_node = l_pos_node.view(batch, num_nodes, 1)
        l_negatives_node = similarity_matrix_nodes_1to2[mask_samples_from_same_repr_node].view(batch, num_nodes, -1)
        l_logits_node = torch.cat((l_positives_node, l_negatives_node), dim=-1).view(batch*num_nodes, -1)

        r_positives_node = r_pos_node.view(batch, num_nodes, 1)
        r_negatives_node = similarity_matrix_nodes_2to1[mask_samples_from_same_repr_node].view(batch, num_nodes, -1)
        r_logits_node = torch.cat((r_positives_node, r_negatives_node), dim=-1).view(batch*num_nodes, -1)

        logits_node = torch.cat((l_logits_node, r_logits_node), dim=0)

        logits_node /= self.temperature

        labels_node = torch.zeros(2 * batch*num_nodes).to(self.device).long()
        loss_node = self.criterion(logits_node, labels_node)

        return loss / (2 * self.batch_size), loss_node / (2 * batch*num_nodes)
