# Score Calcuation

Currently FID score calculation is supported. This is based on the pytorch-fid codebase, with the only change being that the codebase accepts recursive folder structures 
as well.

To run the code between two folders of images, use the code like so : 

```
python fid_score.py   /path/to/folder1 /path/to/folder2
```
