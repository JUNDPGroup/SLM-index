# SLM-index
## How to use
### 1. Required libraries
#### LibTorch
homepage: https://pytorch.org/get-started/locally/ <br> <br>
CPU version: https://download.pytorch.org/libtorch/nightly/cpu/libtorch-cxx11-abi-shared-with-deps-1.4.0%2Bcpu.zip <br>

#### boost
homepage: https://www.boost.org/ <br>

### 2. Change path
Change the path if you do not want to store the datasets under the project's root path.<br> <br>
Constants.cpp
```
const string Constants::RECORDS = "./files/records/";
const string Constants::QUERYPROFILES = "./files/queryprofile/";
const string Constants::DATASETS = "./datasets/";
```
data_generator.py 
```
if __name__ == '__main__':
    distribution, size, skewness, clusters, filename, dim = parser(sys.argv[1:])

    if distribution == 'uniform':
        filename = "datasets/uniform_%d_1_%d_.csv"
        getUniformPoints(size, filename, dim)
    elif distribution == 'normal':
        filename = "datasets/normal_%d_1_%d_.csv"
        getNormalPoints(size, filename, dim)
    elif distribution == 'skewed':
        filename = "datasets/skewed_%d_%d_%d_.csv"
        getSkewedPoints(size, skewness, filename, dim)
    elif distribution == 'cluster':
        filename = "datasets/cluster_%d_%d_%d_.csv"  
        if clusters:
            getClusterPoints(size, filename, dim, clusters)
        else:
            getClusterPoints(size, filename, dim)
```
### 3. Prepare datasets
The uploaded project code contains datasets. If you want to generate them separately, you can use the following command. Among them, the -c parameter in the generation command for the cluster dataset refers to the number of clusters, and its size needs to change with the size of the -s parameter. By default, there are 100 points in one cluster.
```
python data_generator.py -d uniform -s 20000 -n 1 -f datasets/uniform_20000_1_2_.csv -m 2
```
```
python data_generator.py -d skewed -s 20000 -n 4 -f datasets/skewed_20000_4_2_.csv -m 2
```
```
python data_generator.py -d cluster -s 20000 -c 200 -f datasets/cluster_20000_200_2_.csv -m 2
```

### 4. Run
```
make clean
make -f Makefile
./Exp -c 20000 -d uniform -s 1
./Exp -c 20000 -d skewed -s 4
./Exp -c 20000 -d cluster -s 1
./Exp -c 20000 -d ABUS -s 1
./Exp -c 20000 -d HOTEL -s 1
```











