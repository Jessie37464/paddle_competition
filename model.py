import pgl
import paddle.fluid.layers as L
import pgl.layers.conv as conv
import paddle.fluid as F

def get_norm(indegree):
    float_degree = L.cast(indegree, dtype="float32")
    float_degree = L.clamp(float_degree, min=1.0)
    norm = L.pow(float_degree, factor=-0.5) 
    return norm
    

class GCN(object):
    """Implement of GCN
    """
    def __init__(self, config, num_class):
        self.num_class = num_class
        self.num_layers = config.get("num_layers", 1)
        self.hidden_size = config.get("hidden_size", 64)
        self.dropout = config.get("dropout", 0.5)
        self.edge_dropout = config.get("edge_dropout", 0.0)

    def forward(self, graph_wrapper, feature, phase):
        
        for i in range(self.num_layers):

            if phase == "train":
                ngw = pgl.sample.edge_drop(graph_wrapper, self.edge_dropout) 
                norm = get_norm(ngw.indegree())
            else:
                ngw = graph_wrapper
                norm = graph_wrapper.node_feat["norm"]


            feature = pgl.layers.gcn(ngw,
                feature,
                self.hidden_size,
                activation="relu",
                norm=norm,
                name="layer_%s" % i)

            feature = L.dropout(
                    feature,
                    self.dropout,
                    dropout_implementation='upscale_in_train')

        if phase == "train": 
            ngw = pgl.sample.edge_drop(graph_wrapper, self.edge_dropout) 
            norm = get_norm(ngw.indegree())
        else:
            ngw = graph_wrapper
            norm = graph_wrapper.node_feat["norm"]

        feature = conv.gcn(ngw,
                     feature,
                     self.num_class,
                     activation=None,
                     norm=norm,
                     name="output")

        return feature

class GAT(object):
    """Implement of GAT"""
    def __init__(self, config, num_class):
        self.num_class = num_class 
        self.num_layers = config.get("num_layers", 1)
        self.num_heads = config.get("num_heads", 8)
        self.hidden_size = config.get("hidden_size", 8)
        self.feat_dropout = config.get("feat_drop", 0.6)
        self.attn_dropout = config.get("attn_drop", 0.6)
        self.edge_dropout = config.get("edge_dropout", 0.0)

    def forward(self, graph_wrapper, feature, phase):
        if phase == "train": 
            edge_dropout = self.edge_dropout
        else:
            edge_dropout = 0

        for i in range(self.num_layers):
            ngw = pgl.sample.edge_drop(graph_wrapper, edge_dropout) 
            
            feature = conv.gat(ngw,
                                feature,
                                self.hidden_size,
                                activation="elu",
                                name="gat_layer_%s" % i,
                                num_heads=self.num_heads,
                                feat_drop=self.feat_dropout,
                                attn_drop=self.attn_dropout)

        ngw = pgl.sample.edge_drop(graph_wrapper, edge_dropout) 
        feature = conv.gat(ngw,
                     feature,
                     self.num_class,
                     num_heads=1,
                     activation=None,
                     feat_drop=self.feat_dropout,
                     attn_drop=self.attn_dropout,
                     name="output")
        return feature

   
class APPNP(object):
    """Implement of APPNP"""
    def __init__(self, config, num_class):
        self.num_class = num_class
        self.num_layers = config.get("num_layers", 1)
        self.hidden_size = config.get("hidden_size", 64)
        self.dropout = config.get("dropout", 0.5)
        self.alpha = config.get("alpha", 0.1)
        self.k_hop = config.get("k_hop", 10)
        self.edge_dropout = config.get("edge_dropout", 0.0)

    def forward(self, graph_wrapper, feature, phase):
        if phase == "train": 
            edge_dropout = self.edge_dropout
        else:
            edge_dropout = 0
        feature = L.fc(feature, self.hidden_size, name="linear")
        for i in range(self.num_layers):
            res_feature = feature
            feature = L.dropout(
                feature,
                self.dropout,
                dropout_implementation='upscale_in_train')
            feature = L.fc(feature, self.hidden_size, act="relu", name="lin%s" % i)
            feature = res_feature + feature
            feature = L.relu(feature)
            feature = L.layer_norm(feature, "ln_%s" % i)

        feature = L.dropout(
            feature,
            self.dropout,
            dropout_implementation='upscale_in_train')
        
        feature = L.fc(feature, self.num_class, act=None, name="output")

        feature = conv.appnp(graph_wrapper,
            feature=feature,
            edge_dropout=edge_dropout,
            alpha=self.alpha,
            k_hop=self.k_hop)
        return feature


class RESAPPNP(object):
    """Implement of APPNP"""
    def __init__(self, config, num_class):
        self.num_class = num_class
        self.num_layers = config.get("num_layers", 1)
        self.hidden_size = config.get("hidden_size", 128)
        self.dropout = config.get("dropout", 0.2)
        self.alpha = config.get("alpha", 0.1)
        self.k_hop = config.get("k_hop", 20)
        self.edge_dropout = config.get("edge_dropout", 0.01)
        self.feat_dropout = config.get("feat_drop", 0.3)
        self.attn_dropout = config.get("attn_drop", 0.3)

    def forward(self, graph_wrapper, feature, phase):
        if phase == "train": 
            edge_dropout = self.edge_dropout
        else:
            edge_dropout = 0
        feature = L.fc(feature, self.hidden_size, name="linear")
        for i in range(self.num_layers):
            res_feature = feature
            feature = L.dropout(
                feature,
                self.dropout,
                dropout_implementation='upscale_in_train')
            feature = L.fc(feature, self.hidden_size, act="relu", name="lin%s" % i)
            feature = res_feature + feature
            feature = L.relu(feature)
            feature = L.layer_norm(feature, "ln_%s" % i)
            
        feature = L.dropout(
            feature,
            self.dropout,
            dropout_implementation='upscale_in_train')
        ngw = pgl.sample.edge_drop(graph_wrapper, edge_dropout)    
        feature = conv.gat(ngw,
                            feature,
                            16,
                            activation="elu",
                            name="gat_layer_%s" % i,
                            num_heads=8,
                            feat_drop=self.feat_dropout,
                            attn_drop=self.attn_dropout) 
        feature = conv.gat(ngw,
                            feature,
                            16,
                            activation="elu",
                            name="gat_layer_%s" % i,
                            num_heads=8,
                            feat_drop=self.feat_dropout,
                            attn_drop=self.attn_dropout) 
        feature = L.fc(feature, self.num_class, act=None, name="output")

        feature = conv.appnp(graph_wrapper,
            feature=feature,
            edge_dropout=edge_dropout,
            alpha=self.alpha,
            k_hop=self.k_hop)
        return feature

class SGC(object):
    """Implement of SGC"""
    def __init__(self, config, num_class):
        self.num_class = num_class
        self.num_layers = config.get("num_layers", 1)

    def forward(self, graph_wrapper, feature, phase):
        feature = conv.appnp(graph_wrapper,
            feature=feature,
            edge_dropout=0,
            alpha=0,
            k_hop=self.num_layers)
        feature.stop_gradient=True
        feature = L.fc(feature, self.num_class, act=None, bias_attr=False, name="output")
        return feature

 
class GCNII(object):
    """Implement of GCNII"""
    def __init__(self, config, num_class):
        self.num_class = num_class
        self.num_layers = config.get("num_layers", 1)
        self.hidden_size = config.get("hidden_size", 64)
        self.dropout = config.get("dropout", 0.6)
        self.alpha = config.get("alpha", 0.1)
        self.lambda_l = config.get("lambda_l", 0.5)
        self.k_hop = config.get("k_hop", 64)
        self.edge_dropout = config.get("edge_dropout", 0.0)

    def forward(self, graph_wrapper, feature, phase):
        if phase == "train": 
            edge_dropout = self.edge_dropout
        else:
            edge_dropout = 0

        for i in range(self.num_layers):
            feature = L.fc(feature, self.hidden_size, act="relu", name="lin%s" % i)
            feature = L.dropout(
                feature,
                self.dropout,
                dropout_implementation='upscale_in_train')

        feature = conv.gcnii(graph_wrapper,
            feature=feature,
            name="gcnii",
            activation="relu",
            lambda_l=self.lambda_l,
            alpha=self.alpha,
            dropout=self.dropout,
            k_hop=self.k_hop)

        feature = L.fc(feature, self.num_class, act=None, name="output")
        return feature

class res_unimp_large(object):
    def __init__(self, config, num_class):
        self.num_class = num_class
        self.num_layers = config.get("num_layers", 3)
        self.hidden_size = config.get("hidden_size", 128)
        self.out_size=config.get("out_size", 40)
        self.embed_size=config.get("embed_size", 100)
        self.heads = config.get("heads", 2) 
        self.dropout = config.get("dropout", 0.3)
        self.edge_dropout = config.get("edge_dropout", 0.0)
        self.use_label_e = config.get("use_label_e", False)
    
    # 编码输入        
    def embed_input(self, feature):   
        lay_norm_attr=F.ParamAttr(initializer=F.initializer.ConstantInitializer(value=1))
        lay_norm_bias=F.ParamAttr(initializer=F.initializer.ConstantInitializer(value=0))
        feature=L.layer_norm(feature, name='layer_norm_feature_input', 
                                      param_attr=lay_norm_attr, 
                                      bias_attr=lay_norm_bias)
        return feature
    
    # 连同部分已知的标签编码输入（MaskLabel）
    def label_embed_input(self, feature):
        label = F.data(name="label", shape=[None, 1], dtype="int64")
        label_idx = F.data(name='label_idx', shape=[None, 1], dtype="int64")

        label = L.reshape(label, shape=[-1])
        label_idx = L.reshape(label_idx, shape=[-1])

        embed_attr = F.ParamAttr(initializer=F.initializer.NormalInitializer(loc=0.0, scale=1.0))
        embed = F.embedding(input=label, size=(self.out_size, self.embed_size), param_attr=embed_attr )

        feature_label = L.gather(feature, label_idx, overwrite=False)
        feature_label = feature_label + embed
        feature = L.scatter(feature, label_idx, feature_label, overwrite=True)
     
        lay_norm_attr = F.ParamAttr(initializer=F.initializer.ConstantInitializer(value=1))
        lay_norm_bias = F.ParamAttr(initializer=F.initializer.ConstantInitializer(value=0))
        feature = L.layer_norm(feature, name='layer_norm_feature_input', 
                                      param_attr=lay_norm_attr, 
                                      bias_attr=lay_norm_bias)
        return feature
        
    def forward(self, graph_wrapper, feature, phase):
        if phase == "train": 
            edge_dropout = self.edge_dropout
            dropout = self.dropout
        else:
            edge_dropout = 0
            dropout = 0

        if self.use_label_e:
            feature = self.label_embed_input(feature)
        else:
            feature = self.embed_input(feature)
        if dropout > 0:
            feature = L.dropout(feature, dropout_prob=dropout, 
                                    dropout_implementation='upscale_in_train')
        
        #改变输入特征维度是为了Res连接可以直接相加
        feature = L.fc(feature, size=self.hidden_size * self.heads, name="init_feature")


        for i in range(self.num_layers - 1):
            ngw = pgl.sample.edge_drop(graph_wrapper, edge_dropout) 
            from model_unimp_large import graph_transformer, attn_appnp

            res_feature = feature

            feature, _, cks = graph_transformer(str(i), ngw, feature, 
                                             hidden_size=self.hidden_size,
                                             num_heads=self.heads, 
                                             concat=True, skip_feat=True,
                                             layer_norm=True, relu=True, gate=True)
            if dropout > 0:
                feature = L.dropout(feature, dropout_prob=dropout, 
                                     dropout_implementation='upscale_in_train') 
            
            # 下面这行便是Res连接了
            feature = res_feature + feature 
        
        feature, attn, cks = graph_transformer(str(self.num_layers - 1), ngw, feature, 
                                             hidden_size=self.out_size,
                                             num_heads=self.heads, 
                                             concat=False, skip_feat=True,
                                             layer_norm=False, relu=False, gate=True)

        feature = attn_appnp(ngw, feature, attn, alpha=0.2, k_hop=10)

        pred = L.fc(
            feature, self.num_class, act=None, name="pred_output")
        return pred
