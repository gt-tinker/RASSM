
# check if environment variable RASSM_HOME is set
if [ -z "$RASSM_HOME" ]; then
    echo "RASSM_HOME is not set. Please set it to the path of the RASSM repository"
    exit 1
fi

feature=$1

bash $RASSM_HOME/scripts/run-rassm.sh $feature 4 0 0 1
bash $RASSM_HOME/scripts/run-csr.sh $feature
bash $RASSM_HOME/scripts/run-jstream.sh $feature
bash $RASSM_HOME/scripts/run-aspt.sh $feature
bash $RASSM_HOME/scripts/run-csf-us.sh $feature 128 128
bash $RASSM_HOME/scripts/run-csf-uo.sh $feature 128
