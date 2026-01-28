# ML baseline

```sh
uv sync
uv run python train_grp.py
uv run python train_cql.py --data_glob "/data/mjsoul/mahjong_game_record_4p_thr_202[45]*/*.bin.xz" --lr 5e-4 --batch_size 128 --num_workers 12 --gamma 0.97 --alpha 0.1
# trim-plant-10
uv run python train_online.py --load_model cql_model.pth
# xxx
uv run python train_online.py --load_model cql_model.pth --num_workers 12 --num_steps 5000000
```
