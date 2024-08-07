from minibatch_model import *
from util import *
import torch

def train(config: Config, 
          data: Dataset,
          model: GAT | SAGE):
    
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
    sampler = NeighborSampler(config.fanouts)
    model.train()
    timer = Timer()
            
    train_dataloader = DataLoader(data.graph, data.train_mask, sampler, device="cpu", batch_size=config.batch_size, shuffle=True, use_ddp=False, num_workers=get_num_cores())
    with train_dataloader.enable_cpu_affinity(loader_cores=get_list_cores(), compute_cores=get_list_cores(), verbose=True):
        for epoch in range(config.num_epoch):
            timer.start()
            for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
                
                x = data.feat[input_nodes]
                pred = model(blocks, x)
                tlabel = data.label[output_nodes]
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
    if config.model == "sage":
        model = SAGE(in_feats=data.in_feats, hid_feats=config.hid_size, num_layers=config.num_layers, out_feats=data.num_classes)
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
    