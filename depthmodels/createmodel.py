import os
import sys


def create_model(args):
    
    if args.depthmodel == 'depthpro':

        DEPTHPRO_PATH = args.depthmodelrepo
        sys.path.append(os.path.join(DEPTHPRO_PATH, "src"))

        from depthmodels.depthpro import model_depthpro

        checkpoint_path = os.path.join(DEPTHPRO_PATH, "checkpoints", "depth_pro.pt")
        depthpro = model_depthpro(checkpoint_path)

        return depthpro
    
    elif args.depthmodel == 'unidepth':

        ## unidepth
        UNIDEPTH_PATH = args.depthmodelrepo
        sys.path.append(os.path.join(UNIDEPTH_PATH))

        from depthmodels.unidepth import model_unidepth
        unidepth = model_unidepth()

        return unidepth

    elif args.depthmodel == 'midas':
        
        from depthmodels.midas import model_midas
        midas = model_midas()

        return midas

    elif args.depthmodel == 'depthanything':

        ## depthanything
        DEPTHANYTHING_PATH = args.depthmodelrepo
        sys.path.append(DEPTHANYTHING_PATH, 'metric_depth')

        from depthmodels.depthanything import model_depthanything

        checkpoint_path = os.path.join(DEPTHANYTHING_PATH, 'metric_depth', 'checkpoints', 'depth_anything_metric_depth_outdoor.pt')
        depthanything = model_depthanything(checkpoint_path)

        return depthanything

    ## Adabins
    elif args.depthmodel == 'adabins':
        ADABINS_PATH = args.depthmodelrepo
        sys.path.append(ADABINS_PATH)
       
        from depthmodels.adabins import model_adabins

        checkpoint_path=os.path.join(ADABINS_PATH, "pretrained", "AdaBins_kitti.pt")
        adabins = model_adabins(checkpoint_path)

        return adabins
    
    else:
        return None
