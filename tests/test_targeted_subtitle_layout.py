import copy
import unittest
from test_targeted_transcript import fixture, review
from video_knowledge_pipeline.targeted_transcript import normalize_transcript_source, compose_transcript_draft


class SubtitleLayoutContractTests(unittest.TestCase):
    def example(self):
        raw, sources = fixture()
        raw['chapters'][0]['segments'][0]['timings'] = [
            {'text': '正常', 'start_char': 0, 'end_char': 2, 'start': 2, 'end': 2.8},
            {'text': '错词', 'start_char': 2, 'end_char': 4, 'start': 3, 'end': 4}]
        pack = {'pack_payload_sha256': 'p' * 64, 'transcript_edit': normalize_transcript_source(raw, sources)}
        request = review(pack)
        op = {'id': 'subtitle:fixture', 'action': 'split', 'at': 2, 'scope_start': 0, 'scope_end': 8, 'before': '正常错词，继续。', 'after': '正常 ｜ 错词，继续。'}
        chapter = request['review']['chapters'][0]
        chapter['doc']['attrs'] = {'subtitleEdits': [op]}
        chapter['revisions'].append(dict(chapter['revisions'][0], change_id=op['id'], kind='subtitle_structure', subtitle_layout=copy.deepcopy(op)))
        return pack, request

    def test_native_split_is_subtitle_only_and_keeps_original_document(self):
        pack, request = self.example()
        result = compose_transcript_draft(pack, request)
        self.assertFalse(result['media_modified'])
        self.assertEqual(result['review'], request['review'])

    def test_old_drafts_with_no_layout_and_null_default_still_work(self):
        pack, request = self.example()
        chapter = request['review']['chapters'][0]
        chapter['revisions'].pop()
        for attrs in ({}, {'subtitleEdits': None}, {'subtitleEdits': []}):
            chapter['doc']['attrs'] = attrs
            compose_transcript_draft(pack, request)

    def test_out_of_range_bool_duplicate_and_unsupported_actions_are_rejected(self):
        for field, value in [('at', -1), ('at', True), ('at', 999), ('scope_start', 2), ('scope_end', 99), ('action', 'cut_media')]:
            pack, request = self.example()
            request['review']['chapters'][0]['doc']['attrs']['subtitleEdits'][0][field] = value
            with self.assertRaises(ValueError):
                compose_transcript_draft(pack, request)

    def test_layout_cannot_be_saved_as_an_audio_cut_or_without_its_audit(self):
        for mutation in ('cut', 'missing', 'different'):
            pack, request = self.example()
            chapter = request['review']['chapters'][0]
            if mutation == 'cut': chapter['revisions'][-1]['effect'] = 'cut_media'
            if mutation == 'missing': chapter['revisions'].pop()
            if mutation == 'different': chapter['revisions'][-1]['subtitle_layout']['at'] = 3
            with self.assertRaisesRegex(ValueError, 'layout_decision_invalid'):
                compose_transcript_draft(pack, request)

    def test_split_without_exact_native_token_boundary_is_rejected(self):
        pack, request = self.example()
        chapter = request['review']['chapters'][0]
        chapter['doc']['attrs']['subtitleEdits'][0]['at'] = 3
        chapter['revisions'][-1]['subtitle_layout']['at'] = 3
        with self.assertRaisesRegex(ValueError, 'native_time_required'):
            compose_transcript_draft(pack, request)
