



# # 修改后的Makefile
# CC=g++ -O3 -std=c++14 -fopenmp

# SRCS=$(wildcard *.cpp */*.cpp) 
# OBJS=$(patsubst %.cpp, %.o, $(SRCS))

# # GMP配置
# GMP_INCLUDE = -I/usr/include/x86_64-linux-gnu
# GMP_LIB = -L/usr/lib/x86_64-linux-gnu -lgmpxx -lgmp

# TYPE = CPU

# ifeq ($(TYPE), GPU)
#     INCLUDE = -I/home/liuguanli/Documents/libtorch_gpu/include -I/home/liuguanli/Documents/libtorch_gpu/include/torch/csrc/api/include
#     LIB +=-L/home/liuguanli/Documents/libtorch_gpu/lib -ltorch -lc10 -lpthread
#     FLAG = -Wl,-rpath=/home/liuguanli/Documents/libtorch_gpu/lib
# else
#     INCLUDE = -I/home/libtorch/include -I/home/libtorch/include/torch/csrc/api/include
#     LIB +=-L/home/libtorch/lib -ltorch -lc10 -lpthread
#     FLAG = -Wl,-rpath=/home/libtorch/lib
# endif

# INCLUDE += $(GMP_INCLUDE)
# LIB += $(GMP_LIB) -fopenmp

# NAME=$(wildcard *.cpp)
# TARGET=$(patsubst %.cpp, %, $(NAME))

# $(TARGET):$(OBJS)
# 	$(CC) -o $@ $^ $(INCLUDE) $(LIB) $(FLAG)

# %.o:%.cpp
# 	$(CC) -o $@ -c $< -g $(INCLUDE)

# clean:
# 	rm -rf $(TARGET) $(OBJS)


CC=g++ -O3 -std=c++14 -fopenmp

SRCS=$(wildcard *.cpp */*.cpp) 
OBJS=$(patsubst %.cpp, %.o, $(SRCS))

# GMP配置
GMP_INCLUDE = -I/usr/include/x86_64-linux-gnu
GMP_LIB = -L/usr/lib/x86_64-linux-gnu -lgmpxx -lgmp

# Boost.Serialization 配置
BOOST_INCLUDE = -I/usr/include
BOOST_LIB = -L/usr/lib -lboost_serialization

TYPE = CPU
# TYPE = GPU

ifeq ($(TYPE), GPU)
    INCLUDE = -I/home/liuguanli/Documents/libtorch_gpu/include -I/home/liuguanli/Documents/libtorch_gpu/include/torch/csrc/api/include 
    LIB +=-L/home/liuguanli/Documents/libtorch_gpu/lib -ltorch -lc10 -lpthread
    FLAG = -Wl,-rpath=/home/liuguanli/Documents/libtorch_gpu/lib
else
    INCLUDE = -I/home/libtorch/include -I/home/libtorch/include/torch/csrc/api/include -I/usr/local/include
    LIB +=-L/home/libtorch/lib -ltorch -lc10 -lpthread
    LIB +=-L/usr/local/lib -lophelib
    FLAG += -Wl,-rpath=/home/libtorch/lib
    FLAG += -lntl -Wl,--no-as-needed -lntl
endif

# 添加LZ4库配置 - 直接链接库文件
LZ4_LIB = -llz4
# 添加LZ4到总库
LIB += $(LZ4_LIB)

INCLUDE += $(GMP_INCLUDE) $(BOOST_INCLUDE)
LIB += $(GMP_LIB) $(BOOST_LIB) -fopenmp

NAME=$(wildcard *.cpp)
TARGET=$(patsubst %.cpp, %, $(NAME))

$(TARGET):$(OBJS)
	$(CC) -o $@ $^ $(INCLUDE) $(LIB) $(FLAG)

%.o:%.cpp
	$(CC) -o $@ -c $< -g $(INCLUDE)

clean:
	rm -rf $(TARGET) $(OBJS)
    