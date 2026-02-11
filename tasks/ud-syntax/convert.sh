for lan in en es fr ja zh
do
    # python convert.py ${lan}/raw_data/ pos ${lan}/pos/
    # python convert.py ${lan}/raw_data/ relations ${lan}/relations/
    python convert_parsing.py ${lan}/raw_data/ position ${lan}/position/
    # python convert_parsing.py ${lan}/raw_data/ relative_position ${lan}/relative_position/
done