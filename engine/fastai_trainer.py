# encoding: utf-8
"""
@author:  Tako
@contact: takotabak@gmail.com
"""

import logging

import fastai
from fastai.basic_train import Learner
from fastai.basic_data import DataBunch
from fastai.train import fit_one_cycle
from fastai.metrics import accuracy
from utils import LoggingLog


def do_train(
    cfg,
    model,
    train_dl,
    valid_dl,
    optimizer,
    loss_fn,
    metrics=[],
    callbacks: list = [],
):

    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    data_bunch = DataBunch(train_dl, valid_dl)
    learn = Learner(data_bunch, model, loss_func=loss_fn)
    callbacks.append(LoggingLog(learn, "template_model.train"))
    if metrics:
        learn.metrics = metrics
    learn.fit_one_cycle(epochs, cfg.SOLVER.BASE_LR)

