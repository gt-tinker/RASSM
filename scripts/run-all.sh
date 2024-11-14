

feature=$1

bash $RASSM_HOME/scripts/run-rassm.sh $feature 4 0 0 1
bash $RASSM_HOME/scripts/run-csr.sh $feature
bash $RASSM_HOME/scripts/run-jstream.sh $feature
bash $RASSM_HOME/scripts/run-csf-us.sh $feature 128 128
bash $RASSM_HOME/scripts/run-csf-uo.sh $feature 128
bash $RASSM_HOME/scripts/run-aspt.sh $feature

