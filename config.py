from collections import namedtuple

config = {
    "batch_size": 64,
    "bert_lr": 5e-6, 
    "head_lr": 1e-3,
    "model_name": "bert-base-uncased",
    "train_path": "train.json",
    "dev_path": "dev.json",
    "dev_out_path": "dev_labelled.json",
    "ckpt_path": "model.ckpt",
    "test_size": 0.1,
    "sequence_length": 40,
}

# Make config values accessible by attribute, e.g. config.batch_size
Config = namedtuple("Config", config.keys())
config = Config(**config)

UNIQUE_INTENTS = ['SearchCreativeWork', 'PlayMusic', 'RateBook', 'SearchScreeningEvent', 'BookRestaurant', 'AddToPlaylist', 'GetWeather']
INTENT2ID = {intent: i for i, intent in enumerate(UNIQUE_INTENTS)}
ID2INTENT = {i: intent for i, intent in enumerate(UNIQUE_INTENTS)}

UNIQUE_ENTITIES = ['party_size_number', 'cuisine', 'facility', 'service', 'best_rating', 'object_type', 'restaurant_type', 'served_dish', 'current_location', 'playlist',
 'city', 'genre', 'state', 'playlist_owner', 'country', 'object_name', 'restaurant_name', 'spatial_relation', 'movie_type', 'object_select', 'timeRange', 'music_item',
 'object_location_type', 'party_size_description', 'track', 'condition_temperature', 'rating_unit', 'geographic_poi', 'year', 'movie_name', 'sort', 'entity_name',
 'condition_description', 'location_name', 'album', 'poi', 'object_part_of_series_type', 'rating_value', 'artist']
ID2ENTITY = {
    index: entity_typed 
    for i, entity in enumerate(UNIQUE_ENTITIES)
    for entity_typed, index in {f"B-{entity}": 2*i, f"I-{entity}": 2*i+1}.items()
}
ID2ENTITY.update({len(ID2ENTITY): "O"})
ENTITY2ID = {entity: id for id, entity in ID2ENTITY.items()}