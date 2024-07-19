# noise-free degradations with isotropic Gaussian blurs
#python test.py --test_only \
#               --dir_data='./Train/'\
#               --data_test='Random' \
#               --model='DASR' \
#               --scale='4' \
#               --resume=600 \
#               --blur_type='iso_gaussian' \
#               --noise=0.0 \
#               --sig=0.0 \
#               --save_results=True

python test.py --test_only \
               --dir_data='./Train/'\
               --data_test='Random' \
               --model='DMGSR' \
               --scale='4' \
               --resume=584 \
               --blur_type='iso_gaussian' \
               --noise=0.0 \
               --sig=0.0 \
               --save_results=True \
               --save='DMGSR'
#python test.py --test_only \
#               --dir_data='./Train/' \
#               --data_test='Set5' \
#               --model='DASR' \
#               --scale='3' \
#               --resume=600 \
#               --blur_type='iso_gaussian' \
#               --noise=0.0 \
#               --sig=1.6 \
#               --save_results=True
#
#
#
#python test.py --test_only \
#               --dir_data='./Train/' \
#               --data_test='Set5' \
#               --model='DASR' \
#               --scale='3' \
#               --resume=600 \
#               --blur_type='iso_gaussian' \
#               --noise=0.0 \
#               --sig=2.4 \
#               --save_results=True
#
#
#python test.py --test_only \
#               --dir_data='./Train/'\
#               --data_test='Set14' \
#               --model='DASR' \
#               --scale='3' \
#               --resume=600 \
#               --blur_type='iso_gaussian' \
#               --noise=0.0 \
#               --sig=0.8 \
#               --save_results=True
#
#
#python test.py --test_only \
#               --dir_data='./Train/' \
#               --data_test='Set14' \
#               --model='DASR' \
#               --scale='3' \
#               --resume=600 \
#               --blur_type='iso_gaussian' \
#               --noise=0.0 \
#               --sig=1.6 \
#               --save_results=True
#
#
#
#python test.py --test_only \
#               --dir_data='./Train/' \
#               --data_test='Set14' \
#               --model='DASR' \
#               --scale='3' \
#               --resume=600 \
#               --blur_type='iso_gaussian' \
#               --noise=0.0 \
#               --sig=2.4 \
#               --save_results=True
#
#
#python test.py --test_only \
#               --dir_data='./Train/'\
#               --data_test='B100' \
#               --model='DASR' \
#               --scale='3' \
#               --resume=600 \
#               --blur_type='iso_gaussian' \
#               --noise=0.0 \
#               --sig=0.8 \
#               --save_results=True
#
#
#python test.py --test_only \
#               --dir_data='./Train/' \
#               --data_test='B100' \
#               --model='DASR' \
#               --scale='3' \
#               --resume=600 \
#               --blur_type='iso_gaussian' \
#               --noise=0.0 \
#               --sig=1.6 \
#               --save_results=True
#
#
#
#python test.py --test_only \
#               --dir_data='./Train/' \
#               --data_test='B100' \
#               --model='DASR' \
#               --scale='3' \
#               --resume=600 \
#               --blur_type='iso_gaussian' \
#               --noise=0.0 \
#               --sig=2.4 \
#               --save_results=True
#
#
#python test.py --test_only \
#               --dir_data='./Train/'\
#               --data_test='Urban100' \
#               --model='DASR' \
#               --scale='3' \
#               --resume=600 \
#               --blur_type='iso_gaussian' \
#               --noise=0.0 \
#               --sig=0.8 \
#               --save_results=True
#
#
#python test.py --test_only \
#               --dir_data='./Train/' \
#               --data_test='Urban100' \
#               --model='DASR' \
#               --scale='3' \
#               --resume=600 \
#               --blur_type='iso_gaussian' \
#               --noise=0.0 \
#               --sig=1.6 \
#               --save_results=True
#
#
#
#python test.py --test_only \
#               --dir_data='./Train/' \
#               --data_test='Urban100' \
#               --model='DASR' \
#               --scale='3' \
#               --resume=600 \
#               --blur_type='iso_gaussian' \
#               --noise=0.0 \
#               --sig=2.4 \
#               --save_results=True
#
#
#python test.py --test_only \
#               --dir_data='./Train/'\
#               --data_test='M109' \
#               --model='DASR' \
#               --scale='3' \
#               --resume=600 \
#               --blur_type='iso_gaussian' \
#               --noise=0.0 \
#               --sig=0.8 \
#               --save_results=True
#
#
#python test.py --test_only \
#               --dir_data='./Train/' \
#               --data_test='M109' \
#               --model='DASR' \
#               --scale='3' \
#               --resume=600 \
#               --blur_type='iso_gaussian' \
#               --noise=0.0 \
#               --sig=1.6 \
#               --save_results=True
#
#
#
#python test.py --test_only \
#               --dir_data='./Train/' \
#               --data_test='M109' \
#               --model='DASR' \
#               --scale='3' \
#               --resume=600 \
#               --blur_type='iso_gaussian' \
#               --noise=0.0 \
#               --sig=2.4 \
#               --save_results=True


## general degradations with anisotropic Gaussian blurs and noises
#python test.py --test_only \
#               --dir_data='D:/LongguangWang/Data' \
#               --data_test='M109' \
#               --model='DASR' \
#               --scale='2' \
#               --resume=600 \
#               --blur_type='aniso_gaussian' \
#               --noise=10.0 \
#               --theta=0.0 \
#               --lambda_1=0.2 \
#               --lambda_2=4.0
#
#cmd /k