import data.leterrier_spectrin

load_3d = data.leterrier_spectrin.load_3d_2

if __name__ == "__main__":
    import tifffile 
    import montage
    import torch
    data3d = load_3d()
    #data3d = load_3d()
    print("Loaded data.")
    stack = montage.make_stack_multiple(data3d, 25, 25, 64)
    tifffile.imwrite("hax/spectrin_stack.tiff", (torch.stack(stack, 0).permute(0,2,3,1)*255).char().numpy())
