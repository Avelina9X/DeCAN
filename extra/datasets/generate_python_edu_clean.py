import multiprocessing as mp
import argparse

from datasets import load_dataset

BLACKLISTED_REPOS = [
    'PrinceShaji/NoobCalculator',
    'sswadkar/pythoncalculator',
    # 'aCoffeeYin/pyreco',
    'Kuzyashin/my_first_calculator',
    'TalesAmaral/Calculadora-no-python',
    'abhishekyana/Prankulator',
    # 'suryaraj93/LuminarPython',
    # 'Dax246/FIT2004',
    'dzaminedz/Wordlist-Generator',
    # 'maro199111/programming-fundamentals',
    # 'aCoffeeYin/pyreco',
    # 'Theo-Cheynel/centrale',
    # 'VenkateshNarayana/PyPackRecommendationTool',
    '/PyPackMasterData.py',
    # 'aCoffeeYin/pyreco',
    'MathLanca/QuizzPython',
    'bill10/books',
    'river-sneed/guess-a-number',
    'kensotall/alexa_anagram',
    'tkppop/getRandValuesForTesting',
    # 'jerrylee17/Algorithms',
    'rhtrj07/Steve',
    # 'frezafoltran/pinla',
    # 'Dax246/FIT2004',
    # 'AL0FT/Python',
    'jglaser1/TorahBibleCodes',
    'vivek07kumar/Tic-Tac-Toe-GAME',
    'Kloppie5/Project-Exploration',
    # 'saurabhwasule/python-practice',
    'Enscivwy/FizzBuzz',
    # 'aCoffeeYin/pyreco',
    # 'saurabhwasule/python-practice',
    'YoungerCode/level1_programming_katas1',
    '2019jeetdas/MCA---Major-Project',
    'Anderi02/My-projects',
    'MKBV/UwUCalculator',
    'satpathyy/Agriculture-Predictions',
    'aCoffeeYin/pyreco',
    # 'opus49/hangman',
    'vision14/Monopoly',
    'ksgwxfan/climate-parser',
    'MaxiMir/luggageOfKnowledge',
    'PrityanshuSingh/Numbers-Calculations',
    'z-data/practice',
    # 'Metalszorny/TddKatas',
    'jishak13/PythonCourse',
    # 'ComputationalReflection/stypy',
    'vatnid/yiddish-tts',
    'seyoseyoseyo/dijkstra-algorithm',
    'Mazkk13/Python-first-100',
    'GOLDBAUS/add-hc',
    'cordoba14/FOR',
    'alex742/Calculator',
    'catsymptote/stupid_calculator',
    'digital-kid/python',
    'Data-Science-Mar2020-JeetDas/Project-09',
    'TaeUpNext/FlipThatPhone',
    
    'my_first_calc',
    'my-first-calc',
]

BLACKLISTED_PATHS = [
    '/FlowControls/File/fileprogrm.py',
    '/Assignment 3/assignment3_test.py',
    '/Assignment 3/test_file2.py',
    '/server/src/repositories/rate.py',
    '/Fun/calculator.py',
    '/Desktop/modules/wordtonum.py',
    '/app/rhyme_distances.py',
    '/Assignment 3/assignment3_test_alika.py',
    '/Morbit_English_Scorer.py',
    '/src/hangman/dictionary.py',
    '/TddKatasInPython/test/Test/StringCalculatorTests.py',
    '/stypy/sgmc/sgmc_cache/testing/test_programs/benchmark_suite/shedskin/amaze.py',
    '/more code/stupidcode.py',
    'test_calculator.py',
    'hardcoded',
    'test_binary_search_tree.py',
    
    'my_first_calc',
    'my-first-calc',
]

BLACKLISTED_EXACT = [
    'ssarangi/algorithms/leetcode/lru_cache.py'
]

def filter_out( row: dict ):
    repo = row[ 'repo_name' ]
    path = row[ 'path' ]
    full = repo + path
    
    for i in BLACKLISTED_REPOS:
        if i in repo:
            return False

    for i in BLACKLISTED_PATHS:
        if i in path:
            return False

    for i in BLACKLISTED_EXACT:
        if i == full:
            return False

    return True

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--hf_cache_dir',
        type=str,
        required=True,
    )

    parser.add_argument(
        '--destination_repo',
        type=str,
        required=True,
    )

    return parser.parse_args()

def main():
    args = parse_args()

    HF_CACHE_DIR = args.hf_cache_dir
    DESTINATION_REPO = args.destination_repo

    ds_python = load_dataset( 'Avelina/python-edu', cache_dir=HF_CACHE_DIR )
    orig_len = len( ds_python[ 'train' ] )

    ds_python = ds_python.filter( filter_out, num_proc=8 )
    clean_len = len( ds_python[ 'train' ] )

    print( f'Removed {orig_len - clean_len} documents!' )

    if input( f'Type CONFIRM to upload dataset to {DESTINATION_REPO}: ' ) == 'CONFIRM':
        ds_python.push_to_hub( DESTINATION_REPO )
    else:
        print( 'Aborting...' )
    

if __name__ == '__main__':
    main()