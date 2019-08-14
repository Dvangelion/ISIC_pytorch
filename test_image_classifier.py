import glob

#ckpt_list = glob.glob('./models/*.tar')

ckpt_list = ['./models/model.ckpt-001.pth.tar','./models/model.ckpt-048.pth.tar','./models/model.ckpt-050.pth.tar','./models/model.ckpt-051.pth.tar','./models/model.ckpt-052.pth.tar']
ckpt_list.sort(key=lambda x:int(x[-11:-8]))
print(ckpt_list[0][-11:-8])