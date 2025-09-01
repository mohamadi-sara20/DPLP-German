import os
import sys
import re
from shutil import copyfile, rmtree
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog= os.path.basename(__file__),
        description='Trains rst segmenter and parser for the train, dev and test data prepared in a specific path.',
    )
    parser.add_argument('basepath')           # positional argument
    parser.add_argument("-sp", "--skip_prepare", type=bool, default=False, help="Go straight to training. No preparation neede (it is already done)")
    parser.add_argument('-op', "--only_prepare", type=bool, default=False, help='Only prepaer data; exit before training parser or segmenter.')
    parser.add_argument("-n", "--num_matrix", type=int, default=0, help="Number of proj-matrices to check (0: unlimited)")
    parser.add_argument("-m", "--matrix_type", type=str, default="identity", help="Projection matrix type: identity (default), standard")
    parser.add_argument('-ss', "--skip_syntax", type=bool, default=False, help='Skips the syntax parsing step in data preparation (for speed)')
    parser.add_argument('-rm', "--relation_map", type=str, default='', help='Json file with relation mapping; used to reduce relations in .dis files.')
    parser.add_argument("--run-stage1", action="store_true", help="Run only the initial preprocessing steps (0-2).")
    parser.add_argument("--run-stage2", action="store_true", help="Run the final training steps (from step 6 onwards).")

    args = parser.parse_args()
    print(args)

    if args.run_stage1:
        print("--- Running Stage 1: Initial Preprocessing ---")
        for path in (f'{args.basepath}/training',f'{args.basepath}/dev', f'{args.basepath}/test'):
            os.system(f'python3 ger_0_preprocess_rs3.py {path}')
            os.system(f'python2 ger_1_rst2dis.py {path}')
            os.system(f'python3 ger_2_txt.py {path}')
        sys.exit("--- Stage 1 Complete ---")

    if args.run_stage2:
        print("--- Running Stage 2: Final Training ---")
        for path in (f'{args.basepath}/training',f'{args.basepath}/dev', f'{args.basepath}/test'):
             os.system(f'python3 ger_6_conll_rst2merge.py {path}')
    
        relmap = args.relation_map
        if len(relmap) == 0:
            relmap = 'skip'
        for path in (f'{args.basepath}/training',f'{args.basepath}/dev', f'{args.basepath}/test'):
            os.system(f'python3 ger_7_prepare_dis.py {path} {args.relation_map}')

        if not os.path.exists(f"{args.basepath}/model"):
            os.mkdir(f"{args.basepath}/model")

        os.system(f'python2 ger_7_dpvocab.py {args.basepath}')
        os.system(f'python2 discoseg/ger_7_seg_train.py {args.basepath}')
        
        os.system(f'python2 ger_8_learn_projmat.py {args.basepath} -n {args.num_matrix} -m {args.matrix_type}')
        os.system(f'python3 ger_8_rels_extract.py {args.basepath}')
        os.system(f'python2 ger_9_parser_train.py {args.basepath} | tee {args.basepath}/result.txt')
        sys.exit("--- Stage 2 Complete ---")


    # This is the original, unmodified execution path for reference
    for path in (f'{args.basepath}/training',f'{args.basepath}/dev', f'{args.basepath}/test'):
        if not args.skip_prepare:
            os.system(f'python3 ger_0_preprocess_rs3.py {path}')
            os.system(f'python2 ger_1_rst2dis.py {path}')
            os.system(f'python3 ger_2_txt.py {path}')
            os.system(f'python3 ger_3_ner.py {path}')
            if not args.skip_syntax:
                os.system(f'python3 ger_4_txt2parse.py {path}')
            os.system(f'python3 ger_5_txt2conll.py {path}')
            os.system(f'python3 ger_6_conll_rst2merge.py {path}')
    
    if args.only_prepare:
        sys.exit()

    relmap = args.relation_map
    if len(relmap) == 0:
        relmap = 'skip'
    for path in (f'{args.basepath}/training',f'{args.basepath}/dev', f'{args.basepath}/test'):
        os.system(f'python3 ger_7_prepare_dis.py {path} {args.relation_map}')

    if not os.path.exists(f"{args.basepath}/model"):
        os.mkdir(f"{args.basepath}/model")

    os.system(f'python2 ger_7_dpvocab.py {args.basepath}')
    os.system(f'python2 discoseg/ger_7_seg_train.py {args.basepath}')
    
    os.system(f'python2 ger_8_learn_projmat.py {args.basepath} -n {args.num_matrix} -m {args.matrix_type}')
    os.system(f'python3 ger_8_rels_extract.py {args.basepath}')
    os.system(f'python2 ger_9_parser_train.py {args.basepath} | tee {args.basepath}/result.txt')
