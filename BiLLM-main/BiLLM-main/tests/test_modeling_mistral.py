# -*- coding: utf-8 -*-


def test_mistral_model():
    import os
    os.environ['BiLLM_START_INDEX'] = '-1'

    from billm import MistralModel, MistralConfig

    model = MistralModel(MistralConfig(vocab_size=128,
                                       hidden_size=32,
                                       intermediate_size=64,
                                       num_hidden_layers=2,
                                       num_attention_heads=2))
    assert model is not None


def test_bimistral_model():
    import os
    os.environ['BiLLM_START_INDEX'] = '1'

    from billm import MistralModel, MistralConfig

    model = MistralModel(MistralConfig(vocab_size=128,
                                       hidden_size=32,
                                       intermediate_size=64,
                                       num_hidden_layers=2,
                                       num_attention_heads=2))
    assert model is not None



def test_bimistral_lm():
    import os
    os.environ['BiLLM_START_INDEX'] = '1'

    from billm import MistralForCausalLM, MistralConfig

    model = MistralForCausalLM(MistralConfig(vocab_size=128,
                                             hidden_size=32,
                                             intermediate_size=64,
                                             num_hidden_layers=2,
                                             num_attention_heads=2))
    assert model is not None
    assert len(model.model.bidirectionas) > 0


def test_bimistral_seq_clf():
    import os
    os.environ['BiLLM_START_INDEX'] = '1'

    from billm import MistralForSequenceClassification, MistralConfig

    model = MistralForSequenceClassification(MistralConfig(vocab_size=128,
                                                           hidden_size=32,
                                                           intermediate_size=64,
                                                           num_hidden_layers=2,
                                                           num_attention_heads=2))
    assert model is not None
    assert len(model.model.bidirectionas) > 0


def test_bimistral_token_clf():
    import os
    os.environ['BiLLM_START_INDEX'] = '1'

    from billm import MistralForTokenClassification, MistralConfig

    model = MistralForTokenClassification(MistralConfig(vocab_size=128,
                                                        hidden_size=32,
                                                        intermediate_size=64,
                                                        num_hidden_layers=2,
                                                        num_attention_heads=2))
    assert model is not None
    assert len(model.model.bidirectionas) > 0
