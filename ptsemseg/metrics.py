import torch

class runningScore(object):

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = torch.zeros(n_classes, n_classes, device=torch.device('cuda')).long() # [[TN, FP], [FN, TP]]

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = torch.bincount(
            n_class * label_true[mask] +
            label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.view(-1), lp.view(-1), self.n_classes)

    def comput_map(self, label_trues, label_preds, beta=2):
        threshold = torch.arange(50, 100, 5, device=torch.device('cuda')).float() / 100. # [0.5, 0.55, ..., 0.95]
        map_all = torch.tensor([], device=torch.device('cuda'))
        for lt, lp in zip(label_trues, label_preds):
            cm = self._fast_hist(lt.view(-1), lp.view(-1), self.n_classes).float()
            u = cm[0,1] + cm[1,0] + cm[1,1]
            iu = cm[1,1] / u if u > 0 else 1.
            F_u = (cm[0,1] + (beta**2) * cm[1,0] + ((1 + beta)**2) * cm[1,1])
            F_beta_score = (((1 + beta)**2) * cm[1,1]) / F_u if F_u > 0 else 1.
            map = torch.tensor([F_beta_score * (iu > th).float() for th in threshold], device=torch.device('cuda')).float().mean(dim=0, keepdim=True)
            map_all = torch.cat([map_all, map], dim=0)
        return map_all.cpu().numpy()

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix.float()
        acc = torch.diag(hist).sum() / hist.sum()
        acc_cls = torch.diag(hist) / hist.sum(dim=1)
        acc_cls = torch.mean(acc_cls)
        iu = torch.diag(hist) / (hist.sum(dim=1) + hist.sum(dim=0) - torch.diag(hist))
        mean_iu = torch.mean(iu)
        freq = hist.sum(dim=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu.cpu().numpy()))

        return {'Overall Acc: \t': acc.cpu().numpy(),
                'Mean Acc : \t': acc_cls.cpu().numpy(),
                'FreqW Acc : \t': fwavacc.cpu().numpy(),
                'Mean IoU : \t': mean_iu.cpu().numpy(),}, cls_iu

    def reset(self):
        self.confusion_matrix = torch.zeros(self.n_classes, self.n_classes, device=torch.device('cuda')).long()
