__all__ = ['build_model']
supported_models = ['FPN', 'UNET', 'HOURGLASS', 'DB', 'DBCismEvenOddLines', 'MASKSPOTTER', 'DB2']


def build_model(model_name, **kwargs):
    assert model_name in supported_models, f'all supported models is {supported_models}'
    model = eval(model_name)(**kwargs)
    return model
