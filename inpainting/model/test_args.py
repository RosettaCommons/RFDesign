from arguments import get_args 

from icecream import ic 

if __name__ == '__main__':

    args, model_param, loader_param, loss_param = get_args()

    ic(args)
    ic(model_param)
    ic(loader_param)
    ic(loss_param)
