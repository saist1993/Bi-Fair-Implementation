
echo "Dataset employed: adult and celeb"

cd ~/codes/Bi-Fair-Implementation/; python bi_fair_runner.py --log_name equal_odds_simple_non_linear -seed 10 20 30 40 50 --dataset_name  adult -fairness_function equal_odds --model simple_non_linear &> log1 &
cd ~/codes/Bi-Fair-Implementation/; python bi_fair_runner.py --log_name equal_odds_simple_linear -seed 10 20 30 40 50 --dataset_name  adult -fairness_function equal_odds --model simple_linear &> log2 &
cd ~/codes/Bi-Fair-Implementation/; python bi_fair_runner.py --log_name equal_opportunity_simple_non_linear -seed 10 20 30 40 50 --dataset_name  adult -fairness_function equal_opportunity --model simple_non_linear &> log3 &
cd ~/codes/Bi-Fair-Implementation/; python bi_fair_runner.py --log_name equal_opportunity_simple_linear -seed 10 20 30 40 50 --dataset_name  adult -fairness_function equal_opportunity --model simple_linear &> log4 &


cd ~/codes/Bi-Fair-Implementation/; python bi_fair_runner.py --log_name equal_odds_simple_non_linear -seed 10 20 30 40 50 --dataset_name  celeb -fairness_function equal_odds --model simple_non_linear &> log5 &
cd ~/codes/Bi-Fair-Implementation/; python bi_fair_runner.py --log_name equal_odds_simple_linear -seed 10 20 30 40 50 --dataset_name  celeb -fairness_function equal_odds --model simple_linear &> log6 &
cd ~/codes/Bi-Fair-Implementation/; python bi_fair_runner.py --log_name equal_opportunity_simple_non_linear -seed 10 20 30 40 50 --dataset_name  celeb -fairness_function equal_opportunity --model simple_non_linear &> log7 &
cd ~/codes/Bi-Fair-Implementation/; python bi_fair_runner.py --log_name equal_opportunity_simple_linear -seed 10 20 30 40 50 --dataset_name  celeb -fairness_function equal_opportunity --model simple_linear &> log8 &

wait



echo "Dataset employed: dutch and crime"


cd ~/codes/Bi-Fair-Implementation/; python bi_fair_runner.py --log_name equal_odds_simple_non_linear -seed 10 20 30 40 50 --dataset_name  dutch -fairness_function equal_odds --model simple_non_linear &> log1 &
cd ~/codes/Bi-Fair-Implementation/; python bi_fair_runner.py --log_name equal_odds_simple_linear -seed 10 20 30 40 50 --dataset_name  dutch -fairness_function equal_odds --model simple_linear &> log2 &
cd ~/codes/Bi-Fair-Implementation/; python bi_fair_runner.py --log_name equal_opportunity_simple_non_linear -seed 10 20 30 40 50 --dataset_name  dutch -fairness_function equal_opportunity --model simple_non_linear &> log3 &
cd ~/codes/Bi-Fair-Implementation/; python bi_fair_runner.py --log_name equal_opportunity_simple_linear -seed 10 20 30 40 50 --dataset_name  dutch -fairness_function equal_opportunity --model simple_linear &> log4 &


cd ~/codes/Bi-Fair-Implementation/; python bi_fair_runner.py --log_name equal_odds_simple_non_linear -seed 10 20 30 40 50 --dataset_name  crime -fairness_function equal_odds --model simple_non_linear &> log5 &
cd ~/codes/Bi-Fair-Implementation/; python bi_fair_runner.py --log_name equal_odds_simple_linear -seed 10 20 30 40 50 --dataset_name  crime -fairness_function equal_odds --model simple_linear &> log6 &
cd ~/codes/Bi-Fair-Implementation/; python bi_fair_runner.py --log_name equal_opportunity_simple_non_linear -seed 10 20 30 40 50 --dataset_name  crime -fairness_function equal_opportunity --model simple_non_linear &> log7 &
cd ~/codes/Bi-Fair-Implementation/; python bi_fair_runner.py --log_name equal_opportunity_simple_linear -seed 10 20 30 40 50 --dataset_name  crime -fairness_function equal_opportunity --model simple_linear &> log8 &

wait



echo "Dataset employed: dutch and crime"


cd ~/codes/Bi-Fair-Implementation/; python bi_fair_runner.py --log_name equal_odds_simple_non_linear -seed 10 20 30 40 50 --dataset_name  german -fairness_function equal_odds --model simple_non_linear &> log1 &
cd ~/codes/Bi-Fair-Implementation/; python bi_fair_runner.py --log_name equal_odds_simple_linear -seed 10 20 30 40 50 --dataset_name  german -fairness_function equal_odds --model simple_linear &> log2 &
cd ~/codes/Bi-Fair-Implementation/; python bi_fair_runner.py --log_name equal_opportunity_simple_non_linear -seed 10 20 30 40 50 --dataset_name  german -fairness_function equal_opportunity --model simple_non_linear &> log3 &
cd ~/codes/Bi-Fair-Implementation/; python bi_fair_runner.py --log_name equal_opportunity_simple_linear -seed 10 20 30 40 50 --dataset_name  german -fairness_function equal_opportunity --model simple_linear &> log4 &


cd ~/codes/Bi-Fair-Implementation/; python bi_fair_runner.py --log_name equal_odds_simple_non_linear -seed 10 20 30 40 50 --dataset_name  compas -fairness_function equal_odds --model simple_non_linear &> log5 &
cd ~/codes/Bi-Fair-Implementation/; python bi_fair_runner.py --log_name equal_odds_simple_linear -seed 10 20 30 40 50 --dataset_name  compas -fairness_function equal_odds --model simple_linear &> log6 &
cd ~/codes/Bi-Fair-Implementation/; python bi_fair_runner.py --log_name equal_opportunity_simple_non_linear -seed 10 20 30 40 50 --dataset_name  compas -fairness_function equal_opportunity --model simple_non_linear &> log7 &
cd ~/codes/Bi-Fair-Implementation/; python bi_fair_runner.py --log_name equal_opportunity_simple_linear -seed 10 20 30 40 50 --dataset_name  compas -fairness_function equal_opportunity --model simple_linear &> log8 &


cd ~/codes/Bi-Fair-Implementation/; python bi_fair_runner.py --log_name equal_odds_simple_non_linear -seed 10 20 30 40 50 --dataset_name  gaussian -fairness_function equal_odds --model simple_non_linear &> log5 &
cd ~/codes/Bi-Fair-Implementation/; python bi_fair_runner.py --log_name equal_odds_simple_linear -seed 10 20 30 40 50 --dataset_name  gaussian -fairness_function equal_odds --model simple_linear &> log6 &
cd ~/codes/Bi-Fair-Implementation/; python bi_fair_runner.py --log_name equal_opportunity_simple_non_linear -seed 10 20 30 40 50 --dataset_name  gaussian -fairness_function equal_opportunity --model simple_non_linear &> log7 &
cd ~/codes/Bi-Fair-Implementation/; python bi_fair_runner.py --log_name equal_opportunity_simple_linear -seed 10 20 30 40 50 --dataset_name  gaussian -fairness_function equal_opportunity --model simple_linear &> log8 &



echo "Dataset employed: encoded_emoji"
cd ~/codes/Bi-Fair-Implementation/; python bi_fair_runner.py --log_name equal_odds_simple_non_linear -seed 10 20 30 40 50 --dataset_name  encoded_emoji -fairness_function equal_odds --model simple_non_linear &> log5 &
cd ~/codes/Bi-Fair-Implementation/; python bi_fair_runner.py --log_name equal_odds_simple_linear -seed 10 20 30 40 50 --dataset_name  encoded_emoji -fairness_function equal_odds --model simple_linear &> log6 &
cd ~/codes/Bi-Fair-Implementation/; python bi_fair_runner.py --log_name equal_opportunity_simple_non_linear -seed 10 20 30 40 50 --dataset_name  encoded_emoji -fairness_function equal_opportunity --model simple_non_linear &> log7 &
cd ~/codes/Bi-Fair-Implementation/; python bi_fair_runner.py --log_name equal_opportunity_simple_linear -seed 10 20 30 40 50 --dataset_name  encoded_emoji -fairness_function equal_opportunity --model simple_linear &> log8 &