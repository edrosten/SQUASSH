from dataclasses import dataclass, field
from pathlib import Path
import sys
import time
import random
import pprint
from typing import Tuple, List, Optional, Union, Iterator, Iterable, TypeVar


from tqdm import tqdm
import tifffile
import torch
from torch.utils.data import DataLoader
from torch import Tensor
import torchvision


from localisation_data import fwhm_to_sigma, sigma_to_fwhm, GeneralLocalisationDataSet
from git import dirname
import save_ply
from network import GeneralPredictReconstruction

# pylint: disable=too-many-instance-attributes
@dataclass 
class TrainingSegment:
    '''Parameters for training'''
    initial_psf: float=200
    final_psf: float=40
    initial_lr: float=0.001
    final_lr: float=0.001
    lr_step_every: int=1
    psf_step_every: int=3
    epochs: int=100

@dataclass
class DataParameters:
    '''Dataset specific parameters'''
    image_size: int=64
    nm_per_pixel: float=10


@dataclass
class DataParametersXYYZ:
    '''Dataset specific parameters'''
    image_size_xy: int
    image_size_z: int
    nm_per_pixel_xy: float
    z_scale: float 
    

@dataclass 
class NetworkParameters:
    '''Network specific parameters'''
    max_repetition_length: float=3.5
    model_size: int=700
    

# pylint: disable=too-many-instance-attributes
@dataclass 
class TrainingParameters:
    '''Parameters for training'''
    # Training params
    batch_size: int = 10
    schedule: List[TrainingSegment]=field(default_factory=lambda: [TrainingSegment()])
    checkpoint_every: int=10
    validity_weight: float=1.0
    normalize_by_group: bool=False

T = TypeVar("T")
def _expand(iterator: Iterable[Iterable[T]])->Iterable[T]:
    '''Concatenate the inner iterables into one large iterable. I.e. flatten it one level'''
    for i in iterator:
        yield from i

def _normalize_sum(batch: Tensor)->Tensor:
    '''
    Divide each image through by the sum
    Images can be any number of dimensions
    '''
    assert batch.shape[1] == 1, "This needs a rethink if C > 1"
    sum_indices = tuple(range(1, batch.ndim))
    image_sum = batch.sum(sum_indices).reshape(-1, *[1]*len(sum_indices)).expand_as(batch)
    return batch/ (image_sum + 0.0001)


def test_normalize_sum()->None:
    '''test'''
    i1 = _normalize_sum(torch.rand(10,1,9,8))
    for i in i1:
        assert abs(i.sum() - 1) < 1e-5
    i2 = _normalize_sum(torch.rand(10,1,9,8,7))
    for i in i2:
        assert abs(i.sum() - 1) < 1e-5



def _scale_to_1(d: torch.Tensor)->torch.Tensor:
    assert d.shape[1] == 1, "This needs a rethink if C > 1"
    reduce = tuple(range(1, d.ndim))
    expand = torch.Size((d.shape[0], *[1]*len(reduce)))
    return (d - d.amin(reduce).reshape(expand).expand_as(d))/(d.amax(reduce)-d.amin(reduce)).reshape(expand).expand_as(d)



#def _normalize_max(batch: Tensor)->Tensor:
#    '''Divide each image through by the sum'''
#    return batch/ batch.max(3).values.max(2).values.unsqueeze(2).unsqueeze(2).expand(batch.shape)

def _cat_scale_to_1(imgs: Union[Tensor,List[Tensor]], is_valid:Tensor)->Tensor:
    '''Scale input to [0,1], batched, size 1 only, cat them all together'''
    if isinstance(imgs, Tensor):
        imgs=[imgs]

    imgs = [i.squeeze(0) for i in imgs]

    border=2
    width = sum(i.shape[-1] for i in imgs) + border * (len(imgs))
    height = max(i.shape[-2] for i in imgs)
    
    img = torch.zeros(3, height, width)
    #Set bg to red to blue 
    img[2,:,:] = is_valid[0]
    img[0,:,:] = 1-is_valid[0]
    
    h=border//2
    for d in imgs:
        # If they are stacks, do a Z project
        if d.ndim == 4:
            d = d.sum(1)
        assert d.ndim == 3 

        i= ((d - d.min())/(d.max()-d.min())).squeeze(0) # Remove channel dim
        img[0,0:i.shape[0], h:h+i.shape[1]] = i
        img[1,0:i.shape[0], h:h+i.shape[1]] = i
        img[2,0:i.shape[0], h:h+i.shape[1]] = i
        h += i.shape[1] + border

    return img



def _normalized_difference(a:List[Tensor], b:List[Tensor])->List[Tensor]:
    return [ _normalize_sum(i) - _normalize_sum(j) for i,j in zip(a,b)]


def _group_normalized_difference(a:List[Tensor], b:List[Tensor])->List[Tensor]:
    batch = a[0].shape[0]
    assert len(a) == len(b)

    sum_indices = tuple(range(1, a[0].ndim))

    a_sums = [ i.sum(sum_indices) for i in a]
    a_sum = torch.stack(a_sums, 1).sum(1) * len(a)
    assert a_sum.numel() == batch and a_sum.ndim==1

    b_sums = [ i.sum(sum_indices) for i in b]
    b_sum = torch.stack(b_sums, 1).sum(1) * len(a)
    assert b_sum.numel() == batch and b_sum.ndim==1

    nsum = [ ai/(a_sum+.0001).reshape(batch,1,1,1).expand_as(ai) - bi/(b_sum+0.0001).reshape(batch,1,1,1).expand_as(bi) for ai,bi in zip(a,b)]
    return nsum
    

def _group_diff_loss(a:List[Tensor], b:List[Tensor], is_valid:Tensor)->Tensor:
    diff = _group_normalized_difference(a, b)
    diff_loss: Tensor =  torch.stack([(abs(d).mean(tuple(range(1,d.ndim))) * is_valid).mean() for d in diff],0).sum()
    return diff_loss


def _loss_func_l(a:List[Tensor], b:List[Tensor], is_valid:Tensor)->Tensor:
    '''Loss function'''
    diff = _normalized_difference(a, b)
    
    # this should work for 2D and 3D data...
    diff_loss: Tensor =  torch.stack([(abs(d).mean(tuple(range(1,d.ndim))) * is_valid).mean() for d in diff],0).sum()
    return diff_loss

def _diff_loss(a:Union[Tensor,List[Tensor]], b:Union[Tensor,List[Tensor]], is_valid:Tensor)->Tensor:
    
    if isinstance(a, Tensor):
        a=[a]
    if isinstance(b, Tensor):
        b=[b]
    return _loss_func_l(a, b, is_valid)



def _scale_loss(scale: Tensor)->Tensor:
    return scale.log().mean().abs()

def _validity_loss(is_valid: Tensor)->Tensor:
    return (torch.ones_like(is_valid) -is_valid).mean()




def _exponential_step(initial: float, final: float, epoch: float, epochs: float, step_every: float)->float:
    # Simple equation is this:
    #   v1 = initial * b** epoch
    # With naive stepping it becomes:
    #   v2 = initial * b** (epoch//step_every*step_every)
    #
    # Note however that the final v2 is the value of v1 at the 
    # final step, so it needs to be corrected for:
    
    final_step = (epochs-1)//step_every
    b = (final/initial)**(1/final_step)
    return float(initial * b** (epoch//step_every)) #** can return so many different types :/ 






def _compute_segment(epoch:int, epochs: int, 
                     initial_psf: float, 
                     final_psf: float, 
                     psf_step_every: float, 
                     initial_lr: float, 
                     final_lr: float, 
                     lr_step_every: int)->Tuple[float,float]:
    
    fwhm = _exponential_step(initial=initial_psf, final=final_psf, epoch=epoch, epochs=epochs, step_every=psf_step_every)
    lr= _exponential_step(initial=initial_lr, final=final_lr, epoch=epoch, epochs=epochs, step_every=lr_step_every)

    return fwhm, lr


def _compute_schedule(epoch: int,
                      schedule: List[TrainingSegment])->Tuple[float,float,int,int]:

    total_epochs = sum(i.epochs for i in schedule)

    segment = None
    start=0
    for i, segment in enumerate(schedule):
        last = start + segment.epochs-1
        if start <= epoch <= last:
            return *_compute_segment(epoch-start, **vars(segment)), total_epochs, i
        
        start += segment.epochs

    raise ValueError('epoch too large')

def retrain(net: GeneralPredictReconstruction, dataset: GeneralLocalisationDataSet, params: TrainingParameters, subdir: Optional[str]=None)->None:
    '''main'''
    param_txt = pprint.pformat(params).split("\n")
    return _train_dataset_from_network(dataset=dataset, net=net, subdir=subdir, param_txt=param_txt, **vars(params))

# pylint: disable=too-many-arguments
def _train_dataset_from_network(dataset:GeneralLocalisationDataSet, net:GeneralPredictReconstruction, subdir:Optional[str], param_txt:List[str],
        batch_size: int,
        checkpoint_every: int,
        schedule: List[TrainingSegment],
        validity_weight: float,
        normalize_by_group: bool
        )->None:

    _train_dataset(subdir, param_txt,
        batch_size=batch_size,
        checkpoint_every=checkpoint_every,
        schedule=schedule,
        net=net,
        dataset=dataset,
        validity_weight=validity_weight,
        normalize_by_group=normalize_by_group
    )
        


class _FixedPermutationSampler(torch.utils.data.Sampler[int]):
    r"""Samples elements randomly, always in the same order, only once from each augmentation

    Args:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source: GeneralLocalisationDataSet) -> None:
        super().__init__()
        self.data_source = data_source
        self.permutation = list(range(0, len(data_source), data_source.get_augmentations()))
        random.shuffle(self.permutation)

    def __iter__(self) -> Iterator[int]:
        return iter(self.permutation)

    def __len__(self) -> int:
        return len(self.data_source)


_DIFF_LOSS_ALPHA=0.01

# pylint: disable=too-many-arguments
def _train_dataset(subdir:Optional[str], param_txt:List[str],
        batch_size: int,
        checkpoint_every: int,
        schedule: List[TrainingSegment],
        net: GeneralPredictReconstruction,
        dataset: GeneralLocalisationDataSet,
        validity_weight: float,
        normalize_by_group: bool,
        )->None:

    output_dir = Path('log') / dirname
    output_dir.mkdir(exist_ok=subdir is not None, parents=True)

    if subdir is not None:
        output_dir = output_dir / subdir
        output_dir.mkdir()

    torch.save(net.state_dict(), output_dir/"initial_net.zip")

    net.save_model_txt(output_dir/"initial")
    net.save_ply_with_axes(output_dir/"initial.ply")


    if normalize_by_group:
        difference_loss = _group_diff_loss
        normalized_difference = _group_normalized_difference

    else:
        difference_loss = _diff_loss
        normalized_difference = _normalized_difference

    with (output_dir/"loss.txt").open('w') as log_loss:
        
        for l in sys.argv:
            print("ARG ", l, file=log_loss)

        for l in param_txt:
            print("PARAMETER ", l, file=log_loss)


        _, initial_lr, epochs, _ = _compute_schedule(0, schedule)


        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        loader_display = DataLoader(dataset, batch_size=1, sampler=_FixedPermutationSampler(dataset))


        optimizer= torch.optim.Adam(net.parameters(), lr=initial_lr)

        time_per_it: Optional[float] = None

        overall_start_time = time.time()
        print(f"START_TIME {overall_start_time}", file=log_loss)

        running_diff_loss = torch.zeros(1, device=next(net.parameters()).device)

        for epoch in range(epochs):

            start_time = time.time()

            fwhm, lr, _, _  = _compute_schedule(epoch, schedule)


            dataset.set_sigma(fwhm_to_sigma(fwhm))
            print(f"FWHM = {fwhm}")
            fwhm_t = torch.tensor(fwhm, device=dataset[0][0].device)

            for g in optimizer.param_groups:
                g['lr'] = lr


            net.train()

            print(len(loader))
            print("-"*80)


            for batch_no, batch in enumerate(tqdm((loader))):


                optimizer.zero_grad()
                
                #FIXME what about scale2?? Currently no loss
                reconstruction, scale, predicted_sigma, is_valid = net(batch, min_sigma_nm=fwhm_to_sigma(fwhm_t))

                diff_loss = difference_loss(batch, reconstruction, is_valid)
                scale_loss = _scale_loss(scale)
                validity_loss = _validity_loss(is_valid)

                # We need all the images, not just the valid ones, to keep track
                # of what the overall loss is, otherwise it just pushes all the
                # losses to zero.
                all_valid_diff_loss = difference_loss(batch, reconstruction, torch.ones_like(is_valid)).detach()
                running_diff_loss = (running_diff_loss*(1-_DIFF_LOSS_ALPHA) + all_valid_diff_loss*_DIFF_LOSS_ALPHA).detach()

                assert not running_diff_loss.requires_grad
                loss = diff_loss + scale_loss*1e-5 + validity_loss * running_diff_loss * validity_weight

                loss.backward() # type: ignore

                optimizer.step()
                
                # pylint: disable=line-too-long
                print(f"{epoch + batch_no/len(loader)} {loss.item()} {diff_loss.item()} {validity_loss.item()} {running_diff_loss.item()} {epoch} {epochs} {batch_no} {len(loader)} {fwhm} {sigma_to_fwhm(predicted_sigma.mean().item())} {lr} ITERATION", file=log_loss)

            net.eval()
            if epoch % checkpoint_every == 0 or epoch == epochs-1:
                with torch.no_grad():

                    log_dir = output_dir / f'{epoch:05d}'
                    log_dir.mkdir()

                    sigmas = []
                    
                    
                    N = len(next(iter(loader_display)))
                    print(N)
                    images: List[List[torch.Tensor]] = [[] for _ in range(N)]
                    reconstructions: List[List[torch.Tensor]] = [[] for _ in range(N)]
                    differences: List[List[torch.Tensor]] = [[] for _ in range(N)]

                    for i, batch in enumerate(loader_display):
                        if epoch < epochs-1 and i > 20:
                            break

                        reconstruction, scale, sigma, is_valid = net(batch, min_sigma_nm=fwhm_to_sigma(fwhm_t))
                        
                        lossdiff = normalized_difference(batch, reconstruction)
                        
                        original_file = log_dir / f'image-{i:05d}.png'
                        reconstruction_file = log_dir / f'recon-{i:05d}.png'
                        diff_file = log_dir / f'diff-{i:05d}.png'

                        torchvision.utils.save_image(_cat_scale_to_1(batch, torch.ones_like(is_valid)), original_file)
                        torchvision.utils.save_image(_cat_scale_to_1(reconstruction, is_valid), reconstruction_file)
                        torchvision.utils.save_image(_cat_scale_to_1(lossdiff, torch.ones_like(is_valid)), diff_file)
                        sigmas.append(sigma.item())
                        print(f"{epoch} {i} {sigma.item()} OUTPUT_IMAGES", file=log_loss)
                        
                        for i in range(N):
                            images[i].append(_scale_to_1(batch[i].cpu().float()).squeeze(1))
                            reconstructions[i].append(_scale_to_1(reconstruction[i].cpu().float()).squeeze(1))
                            differences[i].append(_scale_to_1(lossdiff[i].cpu().float()).squeeze(1))


                    image_stacks = [ torch.cat(i, 0) for i in images]
                    recon_stacks = [ torch.cat(i, 0) for i in reconstructions]
                    ndiff_stacks = [ torch.cat(i, 0) for i in differences]

                    catted = [torch.cat((i, j, k), -1) for i,j,k in zip(image_stacks, recon_stacks, ndiff_stacks)]
                    for i,c in enumerate(catted):
                        axes = 'TZYX' if c.ndim==4 else 'TYX'
                        tifffile.imwrite(log_dir/f'segment-{i:02d}.tiff', c.numpy(), imagej=True, metadata={'axes': axes})
                        
                    torch.save({
                               'state_dict': net.state_dict(),
                               'epoch': epoch,
                               'optimizer': optimizer.state_dict(),
                               }, log_dir/"network.zip")

                    net.save_ply_with_axes(log_dir/"model.ply")
                    net.save_model_txt(log_dir/"current")

            log_dir = output_dir / f'{epoch:05d}-model_only'
            log_dir.mkdir()
            net.save_model_txt(log_dir/"current")

            net.train()

            end_time = time.time()
            iteration_time = end_time - start_time

            
            print(f"EPOCH_END_TIME {end_time} {overall_start_time-end_time}", file=log_loss)

            if not time_per_it:
                time_per_it = iteration_time
            else:
                alpha = 0.5
                time_per_it = (1-alpha) * time_per_it + alpha * iteration_time

            print(f"Done epoch {epoch} {subdir if subdir else ''}")
            print(f"Time per epoch = {time_per_it:.1f}s")
            remaining = int(time_per_it * (epochs - 1 - epoch))
            print(f"Estimated remaining = {remaining//3600}h {remaining//60%60}m {remaining%60}s")

    points, intensities = (i.detach() for i in net.get_model())
    
    torch.cuda.empty_cache()
    torch.save(net.state_dict(), output_dir/"final_net.zip")
    net.save_model_txt(output_dir/"final")
    net.save_ply_with_axes(output_dir/"final.ply")
    try:
        save_ply.save_pointcloud_as_mesh(output_dir/"final_model_mesh_2.0.ply", points.half(), intensities.half(), 2)
    except RuntimeError as err:
        print("Saving mesh failed", *err.args)
