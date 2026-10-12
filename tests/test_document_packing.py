import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('tokenizers')

from modelchef.disk_data import pack_disk
from modelchef.pipeline import train_tokenizer
from modelchef.run_config import Tokenization


def test_document_packing_keeps_complete_examples_and_masks_boundaries(tmp_path):
    rows = [{'text': 'Question A <think>Combine two numbers.</think> Answer 5.'},
            {'text': 'Question B <think>Subtract one number.</think> Answer 2.'}]
    settings = Tokenization(vocab_size=300, packing='document', add_bos=True)
    tokenizer = train_tokenizer(rows, settings)
    blocks, metadata = pack_disk(rows, tokenizer, settings, 128, 1000, tmp_path, 'train')
    assert len(blocks[0]) == 2
    for i, row in enumerate(rows):
        x, y = blocks[0][i], blocks[1][i]
        expected = [tokenizer.token_to_id('<bos>')] + tokenizer.encode(row['text']).ids + [tokenizer.token_to_id('<eos>')]
        assert x[:len(expected)].tolist() == expected
        assert y[:len(expected)-1].tolist() == expected[1:]
        assert (y[len(expected)-1:] == -100).all()
    assert metadata['skipped_oversized_documents'] == 0
    _, skipped = pack_disk([{'text': 'word ' * 1000}, *rows], tokenizer, settings, 128, 1000, tmp_path, 'test')
    assert skipped['skipped_oversized_documents'] == 1


def test_trimming_padding_preserves_next_token_loss_and_gradients():
    from modelchef.model import Decoder
    from modelchef.pipeline import trim_padded_pair
    from modelchef.run_config import Architecture
    torch.set_num_threads(1)
    torch.manual_seed(4)
    model = Decoder(Architecture(layers=1, hidden_size=32, attention_heads=4,
                                kv_heads=2, intermediate_size=64, context_length=64), 64)
    x, y = torch.randint(64, (2, 64)), torch.randint(64, (2, 64))
    y[:, 20:] = -100
    full = torch.nn.functional.cross_entropy(model(x).flatten(0, 1), y.flatten())
    full.backward()
    gradients = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    tx, ty = trim_padded_pair(x, y)
    assert tx.shape == (2, 20)
    trimmed = torch.nn.functional.cross_entropy(model(tx).flatten(0, 1), ty.flatten())
    trimmed.backward()
    torch.testing.assert_close(full, trimmed)
    for expected, parameter in zip(gradients, model.parameters()):
        torch.testing.assert_close(expected, parameter.grad, rtol=1e-4, atol=1e-5)
