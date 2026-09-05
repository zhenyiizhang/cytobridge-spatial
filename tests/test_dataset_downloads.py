import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from CytoBridge import datasets


def archive_record(tmp_path, name='example_analysis_data.zip', split=True):
    archive = tmp_path / name
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.writestr('data/example/values.csv', 'time,value\n0,1\n')
    blob = archive.read_bytes()
    parts = [blob[:len(blob)//2], blob[len(blob)//2:]] if split else [blob]
    records = []
    for index, payload in enumerate(parts):
        path = tmp_path / f'part{index}'
        path.write_bytes(payload)
        records.append({'name': path.name, 'url': path.as_uri(), 'bytes': len(payload),
                        'sha256': hashlib.sha256(payload).hexdigest()})
    return {'archive': name, 'bytes': len(blob), 'sha256': hashlib.sha256(blob).hexdigest(),
            'parts': records}


def test_multipart_download_extracts_and_reuses_files(tmp_path):
    record = archive_record(tmp_path)
    output = tmp_path / 'reader'
    datasets._download_archive(record, output)
    assert (output / 'data/example/values.csv').read_text() == 'time,value\n0,1\n'
    # Once downloaded, another call does not need the remote archives.
    for part in tmp_path.glob('part*'):
        part.unlink()
    datasets._download_archive(record, output)
    assert not list((output / '.cytobridge').glob('download-*'))


def test_incomplete_download_is_not_extracted(tmp_path):
    record = archive_record(tmp_path)
    record['parts'][0]['sha256'] = '0' * 64
    with pytest.raises(OSError, match='Incomplete download'):
        datasets._download_archive(record, tmp_path / 'reader')
    assert not (tmp_path / 'reader/data').exists()


@pytest.mark.parametrize('name', ['../outside', '/absolute', 'data/../../outside', 'data\\outside'])
def test_archive_cannot_write_outside_output(tmp_path, name):
    archive = tmp_path / 'bad.zip'
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.writestr(name, 'input')
    with pytest.raises(ValueError, match='archive path|Archive path'):
        datasets._extract(archive, tmp_path / 'reader')


def test_download_does_not_overwrite_existing_data(tmp_path):
    record = archive_record(tmp_path)
    output = tmp_path / 'reader'
    destination = output / 'data/example/values.csv'
    destination.parent.mkdir(parents=True)
    destination.write_text('my own analysis')
    with pytest.raises(FileExistsError):
        datasets._download_archive(record, output)
    assert destination.read_text() == 'my own analysis'


def test_api_selects_model_and_analysis_by_default(tmp_path, monkeypatch):
    manifest = tmp_path / 'manifest.json'
    names = ['example_model.zip', 'example_analysis_data.zip', 'example_population_data.zip']
    manifest.write_text(json.dumps({'archives': [{'archive': name} for name in names]}))
    monkeypatch.setattr(datasets, '_MANIFEST', manifest)
    downloaded = []
    monkeypatch.setattr(datasets, '_download_archive', lambda record, path: downloaded.append(record['archive']))
    assert datasets.download('example', tmp_path) == tmp_path / 'data/example'
    assert downloaded == names[:2]
    downloaded.clear()
    datasets.download('example', tmp_path, kind='all')
    assert downloaded == names
    with pytest.raises(ValueError, match='Unknown dataset'):
        datasets.download('not-a-dataset', tmp_path)


def test_published_manifest_has_complete_parts():
    manifest = json.loads(datasets._MANIFEST.read_text())
    names = [item['archive'] for item in manifest['archives']]
    assert len(set(names)) == len(names)
    for item in manifest['archives']:
        assert sum(part['bytes'] for part in item['parts']) == item['bytes']
        assert len(item['sha256']) == 64
        for part in item['parts']:
            assert '/releases/download/paper-data-' in part['url']
            assert 'untagged-' not in part['url']
            assert len(part['sha256']) == 64


def test_public_api_fallback_does_not_use_credentials(monkeypatch):
    calls = []
    sentinel = object()
    def open_url(request, timeout):
        calls.append(request)
        if len(calls) == 1:
            raise TimeoutError('Download hostname unreachable')
        return sentinel
    monkeypatch.setattr(datasets, 'urlopen', open_url)
    assert datasets._open_download({'url': 'https://github.com/example', 'asset_id': 123}) is sentinel
    assert calls[1].full_url.endswith('/releases/assets/123')
    assert calls[1].headers['Accept'] == 'application/octet-stream'
    assert all('Authorization' not in call.headers for call in calls)
