# Pytorch DataLoader for Variable inputs
Use this dataloader to use variable inputs efficiently. It sorts the inputs according to lenghts in-order to minimizing the padding size.

## Example
Without using Sampler 
```
batch_size = 2
x = [[1,2],[3,4,5,6],[4],[1,2,3,4,5,6]]
# outputs will be 
Batch 1: [[1,2,0,0],
          [3,4,5,6]]

Batch 2:[[4,0,0,0,0,0]
         [1,2,3,4,5,6]]

```
With using Sampler
```
x = [[1,2],[3,4,5,6],[4],[1,2,3,4,5,6]]
# outputs will be
Batch 1: [[1,2,3,4,5,6],
          [3,4,5,6,0,0]]

Batch 2: [[1,2]
          [4,0]]
```

With Latter being much more efficient.

## Credits
Sortish sampler code was taken from the Fast.Ai(0.7)
