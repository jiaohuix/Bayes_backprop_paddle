import paddle
import paddle.nn.functional as F

@paddle.no_grad()
def predict(conf,model,test_loader):
    model.eval()
    n_samples=conf['model']['valid_ens']
    res=[]
    for (inputs,_) in test_loader:
        mean_prob = []
        for sample in range(n_samples):
            # forward
            outputs = model(inputs)
            # mean prob
            mean_prob.append(F.softmax(outputs, axis=-1).unsqueeze(2))
        # acc
        mean_prob = paddle.concat(mean_prob, axis=-1)
        mean_prob = paddle.mean(mean_prob, axis=-1)
        pred = paddle.argmax(mean_prob, axis=1).numpy()
        res.append(list(pred))
    return res

