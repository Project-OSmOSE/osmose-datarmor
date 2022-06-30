#!/bin/bash

# $1 -> dataset_id
# $2 -> folder_in
# $3 -> target_fs
# $4 -> ind_min (for qsub jobs)
# $5 -> ind_max (for qsub jobs)
# $6- -> folder_out
# $7- -> new_audioFileDuration
# $8- -> orig_audioFileDuration
# $9- -> nber_segments

# [! -f filename] checks for file existence        

cd "/home/datawork-osmose/dataset/"$1"/raw/audio/"

COUNTER=0
for f in $2/*.wav;

    do

        COUNTSEG=0

        #FULL_FILENAME=$f
        FILENAME=${f##*/}
        FILENAME=${FILENAME%.*}
        
        COUNTER=$[$COUNTER +1]
        
        if [ "$COUNTER" -ge "$4" ] && [ "$COUNTER" -le "$5" ];# && [ ! -f $3"/${FILENAME}" ]  
        
        then         

            if [ $7 -eq $8 ]
            
            then

                /appli/sox/sox-14.4.2_gcc-7.2.0/bin/sox "${f}" -r $3 -t wavpcm $6"/${FILENAME}.wav"
                
                echo $6"/${FILENAME}.wav"

            else                

                for indi in `seq 0 $7 $8`;       

                do

                    if [ $COUNTSEG -eq $9 ]; then 
                        break
                    fi


                    if [ $COUNTSEG -le 9 ]; then
                        /appli/sox/sox-14.4.2_gcc-7.2.0/bin/sox "${f}" -r $3 -t wavpcm $6"/${FILENAME}_seg00${COUNTSEG}.wav" trim $indi $7
                        echo $6"/${FILENAME}_seg00${COUNTSEG}.wav"

                    elif [ $COUNTSEG -le 99 ]; then
                        /appli/sox/sox-14.4.2_gcc-7.2.0/bin/sox "${f}" -r $3 -t wavpcm $6"/${FILENAME}_seg0${COUNTSEG}.wav" trim $indi $7
                        echo $6"/${FILENAME}_seg0${COUNTSEG}.wav"                            

                    else
                        /appli/sox/sox-14.4.2_gcc-7.2.0/bin/sox "${f}" -r $3 -t wavpcm $6"/${FILENAME}_seg${COUNTSEG}.wav" trim $indi $7
                        echo $6"/${FILENAME}_seg${COUNTSEG}.wav"
                        
                    fi

                    COUNTSEG=$[$COUNTSEG +1]
                                        
                done

            fi
                
        fi
                
    done

rm "/home/datawork-osmose/dataset/"$1"/analysis/ongoing_pbsFiles/pbs_resample_"$4".pbs"
