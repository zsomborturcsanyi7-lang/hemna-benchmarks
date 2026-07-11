"""NEURA tokenizer check"""
import sys, os

# Try loading sentencepiece
try:
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(r'C:\NeuraNode\bitnet\data\bitnet_pretrain\tokenizer\tokenizer.model')
    print(f'VOCAB SIZE: {sp.GetPieceSize()}')
    print(f'Elso 10 token: {[sp.IdToPiece(i) for i in range(10)]}')
    print(f'Utolso 5 token: {[sp.IdToPiece(sp.GetPieceSize()-1-i) for i in range(5)]}')
    
    # Test encode/decode
    test = "A mesterseges intelligencia egy csodalatos dolog."
    ids = sp.EncodeAsIds(test)
    decoded = sp.DecodeIds(ids)
    print(f'\nTeszt encode: {test}')
    print(f'Token IDk: {ids}')
    print(f'Visszafejtve: {decoded}')
except Exception as e:
    print(f'Hiba: {e}')
    # Try looking at vocab file
    try:
        with open(r'C:\NeuraNode\bitnet\data\bitnet_pretrain\tokenizer\tokenizer.vocab', 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i < 10:
                    print(f'  {line.strip()}')
    except Exception as e2:
        print(f'Vocab file error: {e2}')
