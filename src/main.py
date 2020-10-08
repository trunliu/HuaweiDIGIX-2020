import numpy as np
from numpy import linalg as LA
import os, shutil
#import matplotlib.pyplot as plt
import time
import datetime
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, Dropout, MaxPooling2D,AveragePooling2D
from keras.models import Sequential
from keras.preprocessing import image
from keras.applications.densenet import DenseNet169
from keras.applications.densenet import preprocess_input as preprocess_input_densenet
from keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
from keras.models import load_model, Model
from keras.callbacks import ReduceLROnPlateau
from keras import optimizers
import keras.backend as K
import csv
import h5py
# 防止报错(OSError: image file is truncated)
from PIL import ImageFile
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True

######################### [调参] ###########################################
dataset_dir = 'test_data_B'              # 测试数据集test文件目录
save_model_name = 'model.h5'             # 保存模型的名称
database_name = 'feature_database.h5'    # 保存特征数据库的名称
submission_dir = 'submission.csv'        # 保存检索结果的名称

rate = 1                                 # 复制数据集的比例
scale = 0.8                              # 训练集和验证集比例
batch_size = 24                          # 训练一批照片的数量
epochs = 100                             # 训练批次
input_size = (224, 224, 3)               # 训练输入照片的大小
Adam_learn_rate = 3e-4                   # 学习率: 网上推荐3e-4
patience = 3                             # 几个epoch调整一次学习率
factor = 0.5                             # 每次调节学习率的比例系数

############################################################################
print("复制原{}%数据集用于训练".format(int(rate * 100)))
print("训练集和验证集比例系数:", scale)
print("batch_size大小:", batch_size)
print("epochs大小:", epochs)
print("Adam学习率:", Adam_learn_rate)
print("每次调节学习率的系数:", factor)
print("调整学习率频率:", patience)

# 官网提供的数据集文件格式
# data
# |--test_data_A
#    |--test_data_A
#       |--gallery
#       |--query
# |--train_data
#    |--DIGIX_000000
#    |--......
#    |--DIGIX_003096
def create_train_test_dataset_folder():
    print("开始复制数据集，本地创建训练/验证/测试集")
    dataset_train_dir = os.path.join(dataset_dir, 'train_data')
    dataset_test_dir = os.path.join(dataset_dir, 'test_data_A\\test_data_A')

    # 当前目录下创建文件夹用来存放整理过后的数据集
    file_names = ['train', 'validation', 'test']
    test_folders = ['query', 'gallery']
    class_names = os.listdir(dataset_train_dir)[:int(3097 * rate)]
    for file in file_names:
        file_dir = os.path.join(file)
        if not os.path.exists(file_dir): os.mkdir(file_dir)
        if file != 'test':
            for class_name in class_names:
                if class_name == 'label.txt': continue
                class_dir = os.path.join(file_dir, class_name)
                if not os.path.exists(class_dir):
                    os.mkdir(class_dir)
        else:
            query_dir = os.path.join(file_dir, test_folders[0])
            gallery_dir = os.path.join(file_dir, test_folders[1])
            if not os.path.exists(query_dir): os.mkdir(query_dir)
            if not os.path.exists(gallery_dir): os.mkdir(gallery_dir)

    # 从官方数据集中(train_data)提取数据构建训练集(data\train)和验证集(data\validation)
    print("训练集/验证集的数量比例: {0} : {1}".format(int(scale * 10), int(10 - scale * 10)))
    for class_name in tqdm(class_names):
        if class_name == 'label.txt': continue
        dataset_class_dir = os.path.join(dataset_train_dir, class_name)
        image_names = os.listdir(dataset_class_dir)
        cnt = len(image_names)
        train_names = image_names[:int(scale * cnt)]
        validation_names = image_names[int(scale * cnt):]
        # 复制用于训练的照片
        for image_name in train_names:
            src = os.path.join(dataset_class_dir, image_name)
            dst = os.path.join(file_names[0], class_name, image_name)
            if not os.path.exists(dst):
                shutil.copyfile(src, dst)
        # 复制用于验证的照片
        for image_name in validation_names:
            src = os.path.join(dataset_class_dir, image_name)
            dst = os.path.join(file_names[1], class_name, image_name)
            if not os.path.exists(dst):
                shutil.copyfile(src, dst)

    # 从官方测试集(test_data_A\test_data_A)中提取数据构建测试集(date\test)
    for test_folder in test_folders:
        dataset_test_path = os.path.join(dataset_test_dir, test_folder)
        image_list = os.listdir(dataset_test_path)
        test_cnt = len(image_list)
        print("{0}集有照片: {1}".format(test_folder, int(rate * test_cnt)))
        cnt = 0
        for image_name in tqdm(image_list):
            src = os.path.join(dataset_test_path, image_name)
            dst = os.path.join(file_names[2], test_folder, image_name)
            if not os.path.exists(dst): shutil.copyfile(src, dst)
            cnt += 1
            if cnt == int(test_cnt * rate): break


# 统计复制了多少数据用于训练的照片(数据增强时需要)
def train_dataset_total_cnt():
    print("统计训练/验证照片数量")
    train_cnt = 0
    val_cnt = 0
    class_names = os.listdir('train')
    for class_name in tqdm(class_names):
        if class_name == 'label.txt': continue
        image_names = os.listdir(os.path.join('train', class_name))
        train_cnt += len(image_names)

    for class_name in tqdm(class_names):
        if class_name == 'label.txt': continue
        image_names = os.listdir(os.path.join('validation', class_name))
        val_cnt += len(image_names)

    print("训练照片数量:", train_cnt)
    print("验证照片数量:", val_cnt)
    return train_cnt, val_cnt

# 自定义网路
class myNet:
    def __init__(self):
        self.input_shape = (input_size[0], input_size[1], input_size[2])
        self.weight = 'imagenet'
        self.pool = 'max'
        self.densenet = DenseNet169(
            weights= self.weight,
            input_shape= (self.input_shape[0], self.input_shape[1], self.input_shape[2]),
            include_top= False
        )
        self.model = self.build()

    # 构造网络
    def build(self):
        self.model = Sequential()
        self.model.add(self.densenet)   # 使用densenet
        self.model.add(Dropout(0.5))
        self.model.add(AveragePooling2D((2, 2), strides=2))
        self.model.add(Flatten())
        self.model.add(Dense(int(3096 * rate) + 1, activation='softmax'))
        return self.model


# 图像增强
def enhenceImage(banch_size):
    print("进行图像增强")
    train_dir = 'train'
    validation_dir = 'validation'
    # 训练数据集增强方式
    train_datagen = ImageDataGenerator(
        rescale= 1. / 255,
        rotation_range= 40,
        width_shift_range= 0.2,
        height_shift_range= 0.2,
        shear_range= 0.2,
        horizontal_flip= True,
        fill_mode= 'nearest'
    )
    # 验证集增加方式
    validation_datagen = ImageDataGenerator(
        rescale= 1. / 255
    )
    # 训练数据集增强
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size= (input_size[0], input_size[1]),
        batch_size= batch_size,
        class_mode='categorical',
        seed= 2020,
        interpolation='box'
    )
    # 验证集增强
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size= (input_size[0], input_size[1]),
        batch_size= batch_size,
        class_mode='categorical',
        seed = 2020,
        interpolation = 'box'
    )
    return train_generator, validation_generator


# 冻结densenet部分卷积层
def frozenNet(net):
    # 解冻所以层
    for layer in net.densenet.layers:
        layer.trainable = True
    print("冻结前:", len(net.densenet.trainable_weights))

    # 冻结前n层数
    n = 590
    for layer in net.densenet.layers[:n]:
        layer.trainable = False
    print("冻结后:", len(net.densenet.trainable_weights))
    return net

# 冻结后要重新编译网络才能生效
def compileNet(net):
    print("重新编译网络")
    net.model.compile(
        loss= 'categorical_crossentropy',
        optimizer= optimizers.Adam(lr = Adam_learn_rate),
        metrics= ['accuracy']
    )
    return net

# 训练网络
def training(net, train_generator, validation_generator, train_cnt, val_cnt):
    print("开始训练")
    start = time.clock()
    # 监控loss通过回调函数实现降低学习率，当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        patience = patience,
        mode= 'auto',
        factor= factor,
    )

    steps_per_epoch = train_cnt// batch_size
    validation_steps = val_cnt // batch_size

    H = net.model.fit_generator(
        train_generator,
        steps_per_epoch= steps_per_epoch,
        epochs= epochs,
        validation_data= validation_generator,
        validation_steps= validation_steps,
        callbacks= [reduce_lr]
    )

    end = time.clock()
    print("训练耗时:{0}s".format(end - start))
    return H

# 保存模型
def save(net, modelName):
    net.model.save(save_model_name)
    print("成功保存模型至路径:", save_model_name)

# 训练结果可视化
# def visualize_result(H):
#     acc = H.history['acc']
#     val_acc = H.history['val_acc']
#     loss = H.history['loss']
#     val_loss = H.history['val_loss']
#     epoch = range(1, len(loss) + 1)
#     fig, ax = plt.subplots(1, 2, figsize = (10, 4))
#     ax[0].plot(epoch, loss, label = 'train_loss')
#     ax[0].plot(epoch, loss, label = 'val_loss')
#     ax[0].set_xlabel('Epoch')
#     ax[0].set_ylabel('Loss')
#     ax[0].legend()
#     ax[1].plot(epoch, acc, label = 'train_acc')
#     ax[1].plot(epoch, val_acc, label = 'val_acc')
#     ax[1].set_xlabel('Epoch')
#     ax[1].set_ylabel('Acc')
#     ax[1].legend()
#     plt.savefig('visual_result.png')
#     plt.show

# 选择训练模型进行预测(提取图像特征)
class extract_model:
    def __init__(self, model_dir):
        self.weight = 'imagenet'
        self.pooling = 'max'
        self.input_shape = (input_size[0], input_size[1], input_size[2])

        # 用自定义得网路预测
        # self.model = load_model(save_model_name)
        # self.model = Model(
        #     inputs= self.model.inputs,
        #     outputs=self.model.get_layer('flatten_1').output
        #     #outputs= self.model.get_layer('dense_1').output
        # )

        # 完全用预训练得模型进行预测
        self.model = DenseNet169(
            weights= self.weight,
            input_shape= (self.input_shape[0], self.input_shape[1], self.input_shape[2]),
            include_top= False,
            pooling = self.pooling
        )

    # 提取图像特征
    def extract_feature(self, img_dir):
        img = image.load_img(
            img_dir,
            target_size=(self.input_shape[0], self.input_shape[1])
        )
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        ima = preprocess_input_densenet(img)
        feat = self.model.predict(img)
        norm_feat = feat[0] / LA.norm(feat[0])
        norm_feat = [i.item() for i in norm_feat]
        return norm_feat

# 为官方提供的测试集中的图库照片(gallery)建立特征库,保存到目录: ./featureDatabase/feature_database.h5
def setup_feature_database():
    save_dir = 'featureDatabase'
    if not os.path.exists(save_dir): os.mkdir(save_dir)

    gallery_dir = dataset_dir + '/gallery'
    gallery_list = [os.path.join(gallery_dir, f) for f in os.listdir(gallery_dir) if f.endswith('.jpg')]

    loaded_model = extract_model(os.path.join(save_model_name))
    loaded_model.model.summary()
    print("成功加载模型并去除全连接层,模型地址:",os.path.join(save_model_name))

    print("开始提取图库特征建立特征库")
    feats = []
    names = []
    for img_dir in tqdm(gallery_list):
        norm_feat = loaded_model.extract_feature(img_dir)
        img_name = os.path.split(img_dir)[1]
        feats.append(norm_feat)
        names.append(img_name)
        pass

    feats = np.array(feats)
    output = os.path.join(save_dir, database_name)
    h5f = h5py.File(output, 'w')
    h5f.create_dataset('dataset_1', data=feats)
    h5f.create_dataset('dataset_2', data=np.string_(names))
    print("成功保存特征库,地址:{}".format(output))
    h5f.close()

# 开始检索官方提供的测试集中的查询图库(query),每张照片返回与它特征最接近的10张图库里的照片
# 结果保存到./submission.csv
def search():
    print("检索查询库，输出查询即结果")
    query_dir = dataset_dir + '/query'
    save_dir = 'featureDatabase'
    feature_database = os.path.join(save_dir, database_name)
    h5f = h5py.File(feature_database, 'r')
    feats = h5f['dataset_1'][:]
    image_name = [imgName.decode('utf-8') for imgName in h5f['dataset_2']]
    h5f.close()

    f = open(submission_dir, 'w', encoding='utf-8', newline="")
    csv_wirter = csv.writer(f)
    loaded_model = extract_model(os.path.join(save_dir, save_model_name))

    # knn: balltree
    knnsearch = KNNSearch(feats)
    for query_name in tqdm(os.listdir(query_dir)):
        query_feat = loaded_model.extract_feature(os.path.join(query_dir, query_name))
        dist, inds = knnsearch.search(query_feat, 10)
        row = [query_name, '{' + image_name[inds[0][0]], image_name[inds[0][1]], image_name[inds[0][2]],
                image_name[inds[0][3]],
                image_name[inds[0][4]], image_name[inds[0][5]], image_name[inds[0][6]], image_name[inds[0][7]],
                image_name[inds[0][8]], image_name[inds[0][9]] + '}']
        csv_wirter.writerow(row)
    f.close()


if __name__ == '__main__':
    # 对官方数据集进行预处理
    create_train_test_dataset_folder()
    # 统计数量，数据增强时要用
    train_cnt, val_cnt = train_dataset_total_cnt()
    # 建立网络
    net = myNet()
    # 冻结部分卷积层
    net = frozenNet(net)
    # 显示网络结构
    net.model.summary()
    # 重新编译网络
    net = compileNet(net)
    # 通过数据增强来训练网络
    train_generator, validation_generator = enhenceImage(batch_size)
    # 开始训练
    H = training(net, train_generator, validation_generator, train_cnt, val_cnt)
    # 保存训练模型
    save(net, save_model_name)
    # 训练结果可视化
    visualize_result(H)
    # 提取图库特征建立数据库
    setup_feature_database()
    # 检索查询库，输出查询即结果到csv
    search()


