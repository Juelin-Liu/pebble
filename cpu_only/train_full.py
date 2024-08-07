from full_model import GAT, GCN, evaluate, test
from util import Timer, Config, Dataset, get_args, load_dataset
import torch

def train(config: Config, 
          data: Dataset,
          model: GAT | GCN):
    
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
    timer = Timer()
    # training loop
    for epoch in range(config.num_epoch):
        
        timer.start()
        
        model.train()
        logits = model(data.graph, data.feat)
        pred = logits[data.train_mask]
        tlabel = data.label[data.train_mask]
        loss = loss_fcn(pred, tlabel)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = evaluate(config, data, model) if config.eval else 0.0
        
        epoch_time = timer.stop()
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | Epoch Time {:.4f}".format(
                epoch, loss.item(), acc, epoch_time
            ), flush=True
        )

def main():
    config = get_args()
    data = load_dataset(config)
    model = None
    if config.model == "gcn":
        model = GCN(in_feats=data.in_feats, hid_feats=config.hid_size, num_layers=config.num_layers, out_feats=data.num_classes)
    elif config.model == "gat":
        model = GAT(in_feats=data.in_feats, hid_feats=config.hid_size, num_layers=config.num_layers, out_feats=data.num_classes, num_heads=config.num_head)
    else:
        print("unsupported model type", config.model)
        exit(-1)
        
    train(config, data, model)
    test_acc = test(config, data, model)
    print("Test Accuracy {:.4f}".format(test_acc))
    
    
if __name__ == "__main__":
    main()
    