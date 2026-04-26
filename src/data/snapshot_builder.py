#%%
import pandas as pd
import json

FILES_OWNERS = [
    ("../../EnrichedConnectionFiles/enriched_Aleksandar.json", "Aleksandar"),
    ("../../EnrichedConnectionFiles/enriched_Jelena.json", "Jelena"),
    ("../../EnrichedConnectionFiles/enriched_Mihailo.json", "Mihajlo"),
    ("../../EnrichedConnectionFiles/enriched_Petar.json", "Petar")
]

def snapshot_builder():
    seen_profiles = set()
    snapshot = []

    for file, owner in FILES_OWNERS:
        df = pd.read_json(file)
        for row in df.iterrows():
            url = row[1]['source_row']['url']
            if url not in seen_profiles:
                seen_profiles.add(url)
                connection = row[1].to_dict()
                connection['owners'] = [owner]
                snapshot.append(connection)
            else:
                for connection in snapshot:
                    if connection['source_row']['url'] == url:
                        connection['owners'].append(owner)

    json.dump(snapshot, open('../../data/snapshot.json', 'w'), indent=4)


if __name__ == '__main__':
    snapshot_builder()