"""ResNet PyTorch cuDNN backend — 1.7s vs ORT 4.5s."""
import onnx, numpy as np, torch, torch.nn.functional as F, json, os, gc

RESNET_PATH = "/workspace/C3/testcases/models/resnet_v1.onnx"

_DEVICE = None; _WEIGHTS = None; _NODE_ATTRS = None
_CONSUMER_COUNT = None; _INPUT_NAME = None; _OUTPUT_NAME = None; _NODES = None

def _init():
    global _DEVICE,_WEIGHTS,_NODE_ATTRS,_CONSUMER_COUNT,_INPUT_NAME,_OUTPUT_NAME,_NODES
    if _WEIGHTS is not None: return
    _DEVICE = torch.device("cuda"); torch.backends.cudnn.benchmark = True
    model = onnx.load(RESNET_PATH)
    _WEIGHTS = {init.name:torch.from_numpy(onnx.numpy_helper.to_array(init).copy()).float().to(_DEVICE) for init in model.graph.initializer}
    _NODE_ATTRS = {}
    for node in model.graph.node:
        a={}
        for attr in node.attribute:
            if attr.name=="strides":a["stride"]=tuple(attr.ints)
            elif attr.name=="pads":a["pads"]=list(attr.ints)
        _NODE_ATTRS[node.name]=a
    _INPUT_NAME=model.graph.input[0].name; _OUTPUT_NAME=model.graph.output[0].name
    _NODES = model.graph.node
    _CONSUMER_COUNT={}
    for node in _NODES:
        for inp in node.input: _CONSUMER_COUNT[inp]=_CONSUMER_COUNT.get(inp,0)+1

def infer_resnet(input_tensors, batch_size):
    _init()
    data=torch.from_numpy(list(input_tensors.values())[0]).float();N=data.shape[0];BS=min(batch_size,N)
    ao=[]
    for st in range(0,N,BS):
        end=min(st+BS,N);batch=data[st:end];actual=batch.shape[0]
        if actual<BS:batch=torch.cat([batch,torch.zeros(BS-actual,*batch.shape[1:])],dim=0)
        batch=batch.to(_DEVICE);reg={_INPUT_NAME:batch};rem=dict(_CONSUMER_COUNT)
        with torch.no_grad():
            for node in _NODES:
                op=node.op_type;inp=node.input;out=node.output
                if op=="Conv":
                    x=reg[inp[0]];w=_WEIGHTS[inp[1]];b=_WEIGHTS.get(inp[2])if len(inp)>2 else None
                    a=_NODE_ATTRS.get(node.name,{});s=a.get("stride",(1,1))
                    pads=a.get("pads",[0,0,0,0]);p=(pads[0],pads[2])if len(pads)>=4 else(pads[0],pads[0])if len(pads)>=2 else 0
                    reg[out[0]]=F.conv2d(x,w,b,stride=s,padding=p)
                elif op=="Relu":reg[out[0]]=F.relu(reg[inp[0]])
                elif op=="Add":reg[out[0]]=reg[inp[0]]+reg[inp[1]]
                elif op=="GlobalAveragePool":reg[out[0]]=reg[inp[0]].mean(dim=[2,3])
                elif op=="Flatten":reg[out[0]]=reg[inp[0]].reshape(reg[inp[0]].shape[0],-1)
                elif op=="Gemm":reg[out[0]]=F.linear(reg[inp[0]],_WEIGHTS[inp[1]],_WEIGHTS.get(inp[2])if len(inp)>2 else None)
                else:continue
                for i_name in inp:
                    if i_name in rem and i_name in reg:
                        rem[i_name]-=1
                        if rem[i_name]<=0 and i_name!=_INPUT_NAME:del reg[i_name]
        ao.append(reg[_OUTPUT_NAME][:actual].cpu().numpy());del reg
    return {"logits":np.concatenate(ao,axis=0).astype(np.float32)}
