import os


def run_exp(init='stackofstars',device='0',acc_fac=10, res=120,rec_lr=0.001,lr=0.1,batch_size=4,num_epochs=120,wacc=1e-1,wvel=1e-1,fmaps=32
            ,depth=26,resolution_degrading=3,TrajLearning=True,name='',pps=300,wi=100, train=True):
    test_name = f'{init}/res_{res}_accFactor_{acc_fac}_lr_{rec_lr}_sublr_{lr}_TrajLearn_{TrajLearning}_wacc{wacc}_wvel{wvel}_' \
                f'pps_{pps}_wtincr_epoch_{wi}_epochs_{num_epochs}_{name}'
    if train==True:
        if TrajLearning==False:
            os.system(f'CUDA_VISIBLE_DEVICES={device} python3 train.py --test-name={test_name} --acceleration-factor={acc_fac}'
                      f'  --sub-lr={lr}  --initialization={init} --batch-size={batch_size}  --lr={rec_lr}'
                      f' --num-epochs={num_epochs} --G-max=40 --S-max=200 --acc-weight {wacc} --vel-weight {wvel}'
                      f' --resolution_degrading={resolution_degrading} --resolution={res}'
                      f' --f_maps={fmaps} --depth={depth}  --points-per-shot={pps} --trajectory-learning --weight-increase-epoch={wi}')
        else:
            os.system(f'CUDA_VISIBLE_DEVICES={device} python3 train.py --test-name={test_name} --acceleration-factor={acc_fac}'
                      f'  --sub-lr={lr}  --initialization={init} --batch-size={batch_size}  --lr={rec_lr}'
                      f' --num-epochs={num_epochs} --G-max=40 --S-max=200 --acc-weight {wacc} --vel-weight {wvel}'
                      f' --resolution_degrading={resolution_degrading} --resolution={res}'
                      f' --f_maps={fmaps} --depth={depth} --points-per-shot={pps} --weight-increase-epoch={wi}')

    os.system(f'CUDA_VISIBLE_DEVICES={device} python3 reconstructe.py --test-name={test_name}'
              f' --resolution={res} --depth={depth} --resolution-degrading={resolution_degrading}')

    os.system(f'CUDA_VISIBLE_DEVICES={device} python3 evaluate.py --test-name={test_name}'
              f' --resolution-degrading={resolution_degrading}')
