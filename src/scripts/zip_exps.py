import sys
sys.path.append("/home/issam/Research_Ground/FishCount")

from src import mlkit


id2name = {"bd89c1f7f46f9fa19a5af03629df701d":"clf",
            "e61c627e71e649903f6b1689cbf73a52": "reg",
            "6fbd655a5e2279c45965e1ce86d59a3e":"loc",
            "736ab64a07ac9974dd45b8dccc2a6c83":"seg"}


def zip_exps(exp_id_list, savedir_base, outdir_base):
    for exp_id in exp_id_list:
        
        src_dirname = "%s/%s" % (savedir_base, exp_id)
        out_fname = "%s/%s.zip" % (outdir_base, id2name[exp_id])
        mlkit.zipdir(src_dirname, out_fname)
        print("Zipped: %s" % out_fname)

if __name__ == "__main__":
    savedir_base = "/mnt/datasets/public/issam/prototypes/underwater_fish/borgy"
    outdir_base = "/mnt/datasets/public/issam/prototypes/underwater_fish/zipped"
    
    exp_id_list = ["e61c627e71e649903f6b1689cbf73a52",
      "6fbd655a5e2279c45965e1ce86d59a3e",
      "736ab64a07ac9974dd45b8dccc2a6c83"]
    zip_exps(exp_id_list,
             savedir_base=savedir_base,
             outdir_base=outdir_base)