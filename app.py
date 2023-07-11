import torch 
from src.modules.ui.ui_manage import UIManage

if __name__ == "__main__":
    uiManage = UIManage()
    uiManage.ui_full()
    # z = torch.tensor([1, 3])
    # print("z.size:", z.size())

    # a = torch.tensor([[[1, 2], 
    #                    [0, 3], 
    #                    [2, 4]],
    #                    [[1, 2], 
    #                    [0, 3], 
    #                    [2, 4]]])
    # print("a:", a)
    # print("a.size:", a.size())

    # b = a.expand(2, -1, -1, -1)
    # print("b:", b)
    # print("b.size:", b.size())