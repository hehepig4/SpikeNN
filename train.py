from model import MNISTClassifierANN,MNISTClassifierSNN
import torch
import torchvision
import torch.nn as nn
from tqdm.auto import tqdm
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
#Spike Network




if __name__=='__main__':
    t_length=50
    def Copy_encoder(x,t_length=t_length):
        return x.repeat(t_length,1,1,1)
    train_set=torchvision.datasets.MNIST('./data/', train=True, download=True,
                                    transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Lambda(lambda x:x/255.0),
                                        torchvision.transforms.Lambda(lambda x:Copy_encoder(x,t_length))
                                    ]))
    test_loader =torchvision.datasets.MNIST('./data/', train=False, download=True,
                                    transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Lambda(lambda x:x/255.0),
                                        torchvision.transforms.Lambda(lambda x:Copy_encoder(x,t_length))
                                    ]))



    summary=SummaryWriter()
    classifier=MNISTClassifierSNN(torch.zeros([1,t_length,1,28,28]),batch_first=True)

    epochs=30

    train_loader=torch.utils.data.DataLoader(train_set,batch_size=1024,shuffle=True)
    test_loader=torch.utils.data.DataLoader(test_loader,batch_size=1024,shuffle=True)
    crit=nn.CrossEntropyLoss()
    opt=torch.optim.Adam(classifier.parameters(),lr=0.0001)
    accelerator = Accelerator(split_batches=True)
    #accelerator.init_trackers()
    classifier, opt, train_loader, test_loader = accelerator.prepare(classifier, opt, train_loader, test_loader)
    for epoch in range(epochs):
        #accelerator.wait_for_everyone()
        with tqdm(total=len(train_loader),disable=not accelerator.is_local_main_process) as pbar:
            for i,(x,y) in enumerate(train_loader):
                #x,y=x.cuda(),y.cuda()
                out=classifier(x)
                out=out.sum(1)
                out=torch.nn.functional.normalize(out,dim=1,p=1)
                opt.zero_grad()
                loss=crit(out,y)
                accelerator.backward(loss)
                # for name, parms in classifier.named_parameters():	
                #     print('-->name:', name, '-->grad_requirs:',parms.requires_grad, ' -->grad_value:',parms.grad)
                opt.step()
                #print(loss.item(),y[0],out.argmax(1)[0],out[0].sum())
                if accelerator.is_local_main_process:
                    acc=(out.argmax(1)==y).float().mean()
                    pbar.set_description(f"Epoch {epoch} loss {loss.item()} acc {acc.item()}")
                    summary.add_scalar('loss',loss.item(),epoch*len(train_loader)+i)
                    summary.add_scalar('acc',acc.item(),epoch*len(train_loader)+i)
                pbar.update(1)
        
        if accelerator.is_local_main_process:
            with torch.no_grad():
                for i,(x,y) in enumerate(test_loader):
                    #x,y=x.cuda(),y.cuda()
                    out=classifier(x)
                    out=out.sum(1)
                    out=torch.nn.functional.normalize(out,dim=1,p=1)
                    loss=crit(out,y)
                    acc=(out.argmax(1)==y).float().mean()
                    summary.add_scalar('test_loss',loss.item(),epoch*len(test_loader)+i)
                    summary.add_scalar('test_acc',acc.item(),epoch*len(test_loader)+i)
        
    accelerator.wait_for_everyone()
    #accelerator.end_training()
    unwrapped_model = accelerator.unwrap_model(classifier)
    accelerator.save(unwrapped_model.state_dict(), 'model.pth')
    #load:
    #model.load_state_dict(torch.load('model.pth'))