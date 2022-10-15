import torch


def ssn_loss_CE(weight=None, ignore_index=-1):

    ce = torch.nn.CrossEntropyLoss(
        weight=weight, reduction="none", ignore_index=ignore_index
    )

    def fun(y_true, y_pred):
        mc_samples = torch.tensor(y_pred.shape[1])
        batch_size = y_pred.shape[0]
        shape = y_pred.shape[-3:]

        y_true = torch.tile(y_true.unsqueeze(1), (1, mc_samples, 1, 1))

        y_true = torch.reshape(
            y_true, [mc_samples * batch_size, shape[-1] * shape[-2]])
        y_pred = torch.reshape(
            y_pred, [mc_samples * batch_size, shape[-3], shape[-1] * shape[-2]]
        )

        log_prob = -ce(target=y_true, input=y_pred)
        log_prob = log_prob.reshape(
            [batch_size, mc_samples, shape[-1] * shape[-2]])

        log_prob = torch.sum(log_prob, dim=(-1))

        loglikelihood = torch.mean(
            torch.logsumexp(
                log_prob,
                dim=1)) - torch.log(mc_samples)

        loss = -loglikelihood

        return loss

    return fun
