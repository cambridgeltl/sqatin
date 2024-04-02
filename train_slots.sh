for fold in 0 1 2 3 4 5 6 7 8 9
do
        for template in none_none_none usersaid_QUESTION_none
        do
                for domain in banking hotels
                do
                        python turn_data_to_train.py --domain $domain --setting 10 --fold $fold --language english --template_name $template --task slots --train
                        python turn_data_to_train.py --domain $domain --setting 10 --fold $fold --language english --template_name $template --task slots
                        python flan_finetune.py --fold $fold --template_name $template --language english --domain $domain --setting 10 --task slots
                done
        done
done

rm multi3nlu/english/train_*
rm multi3nlu/english/test_*