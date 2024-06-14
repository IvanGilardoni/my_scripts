# script to join second rows of min_point, parameters, values files respectively
# keep first row of a file as title

#path='fm_ref_step1000_sincos'

# in this way, you can pass the folder name in command line with -n 'folder_name'
while getopts n: flag
do
    case "${flag}" in
        n) path=${OPTARG};;
    esac
done

for name in 'min_ff_pars' 'min_Karplus_pars' 'min_lambdas' 'parameters' 'values'
do
    # if existing (to be added)
    first_file=( ${path}/${name}_* )
    l=$first_file
#    printf ${l}
    awk 'FNR==1{print;nextfile}' ${l} > ${path}/${name}_title

    awk 'FNR==2{print;nextfile}' ${path}/${name}_* > ${path}/${name}_
    
    cat ${path}/${name}_title ${path}/${name}_ > ${path}/${name} 
    rm ${path}/${name}_*
done

for name in 'test_contraj' 'test_obs'
do
    if [ ! -f ${path}/${name} ]
    then
        cat ${path}/${name}_* > ${path}/${name}
        rm ${path}/${name}_*
    fi
done