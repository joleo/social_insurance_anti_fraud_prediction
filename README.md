# social_insurance_anti_fraud_prediction
人工智能社会保险反欺诈分析

## 赛题背景

### 整体背景

我国社会保险数据规模巨大，现已覆盖数亿参保人群，内容包罗养老、失业、医疗、工伤、生育等方面，具有典型的大数据特征，将成为智慧医疗发展创新的重要基础。 

近年来, 随着全民医保的基本实现以及即时结算等多项工作的推进, 医疗保险服务的形势也出现了新特点, 欺诈骗保等现象有所增多. 人社部门通过利用数据比对, 诊疗规则筛查等手段取得了一定成效, 但数据应用方式还较为传统, 在数据应用理念, 方法, 技术等方面存在进一步改进和提升的空间。

此项竞赛旨在推动大数据, 人工智能等技术在基本医疗保险医疗服务智能监控领域的应用，提高医保智能监控的针对性和有效性, 从而全面提升社会保障服务的能力和水平, 形成一批可借鉴、可复制、可推广的应用模式和建设方案。

### 业务背景

当下, 针对医疗保险的欺诈骗保日趋增多的现象, 人社部门通过数据比对, 诊疗规则筛查等手段, 加强了对门诊, 住院, 购药等各类医疗服务行为的监控, 但仍无法全面有效地遏制此类行为的发生. 此项竞赛旨在通过大数据, 人工智能等技术变革智慧医疗监控体系, 提高精准识别欺诈骗保的技术能力和水平, 提升医疗保险服务自动化效率, 从而构建更充分, 更全面, 更平衡的社会保障体系。


## 交叉特征
对于购物来说，不同的性别、年龄、职业可能会对不同的商品品牌、商品价格等属性感兴趣，因此我们可以考虑构造这些交叉特征的转化率特征来刻画这种现象。在代码文件`static_features.py`中，我将提取这类特征封装成一个类，可以直接使用。在计算转化率时有些特征组合出现的频次少，我直接将这些频次少的组合进行了过滤，以防止过拟合。另外计算训练集的转化率时，为了防止数据泄露，我采用了五折交叉的方式，将所有数据分成5份，使用其中的4份来计算剩下的那份数据的转化率。

# 组合特征

我们可以想到，同样性别的用户，如果年龄不同、职业不同、消费水平不同，等等，那他们的个人属性肯定也会有区别。所以有必要对这些特征进行组合，比如性别为`1`，年龄为`3`，那么通过组合，我们可以得到一个`性别-年龄`的特征`1_3`。如果使用线性模型，这样做显然是有用的，这样等于在one-hot之后加入了一个非线性特征，但是对树形模型有没有用呢，毕竟树形模型能在训练时学习到非线性特征，我觉得还是有用的，因为树形模型在训练时对特征的组合具有随机性，通过手动构造这样的组合特征等于我们直接的指定要模型学习这种组合特征。
   
    def gen_user_feat():
        """
        组合用户的性别、年龄、职业、星级属性
        """
        user = pd.read_csv("./data/user_file.csv",header=0)
        user['gender_age'] = user['user_gender_id'].astype('str') + '_' + user['user_age_level'].astype('str')
        user['gender_occp'] = user['user_gender_id'].astype('str') + '_' + user['user_occupation_id'].astype('str')
        user['gender_star'] = user['user_gender_id'].astype('str') + '_' + user['user_star_level'].astype('str')
        user['age_occp'] = user['user_age_level'].astype('str') + '_' + user['user_occupation_id'].astype('str')
        user['age_star'] = user['user_age_level'].astype('str') + '_' + user['user_star_level'].astype('str')
        user['occp_star'] = user['user_occupation_id'].astype('str') + '_' + user['user_star_level'].astype('str')
    
        for col in ['gender_age','gender_occp','gender_star','age_occp','age_star','occp_star','user_age_level',\
                    'user_occupation_id','user_star_level']:
            user[col] = lbl.fit_transform(user[col])
            user[col] = remove_lowcase(user[col])

        return user
        
# 时间特征

就是各种特征的时间差分、滑窗统计，不一一细说了。

## 调参
要解决这个问题要么平衡数据，要么就是先确定回归决策树每个叶子结点最小的样本数(min_samples_leaf),再确定分裂所需最小样本数（min_samples_split），才能确定最大深度,这样就能保证不会出现某棵树通过一个feature将数量较少的的正类以较过拟合的简单浅层树拟合出来，而是优先保证了每一次我构造树都尽可能的平衡满足了数据量合理，数据具有样本具有代表性，不会过拟合这样的假设。所以，可以优化为：

- 先确定快速训练的n_estimators和learning_rate，之后所有的调参基于这个确定的值
- 再确定合适的subsample
- 再组合调优最大树深度（max_depth）和叶节点最小样本数（min_samples_leaf）
- 再调优最大叶节点数（max_leaf_nodes）
- 再考虑是否有必要修改叶节点最小权重总值（min_weight_fraction_leaf）,这边是不一定使用的
- 再组合优化分裂所需最小样本数（min_samples_split）
- 最后，优化分裂时考虑的最大特征数（max_features）
- 组合调整n_estimators和learning_rate

lgb参数,https://blog.csdn.net/qq_23069955/article/details/80611701

