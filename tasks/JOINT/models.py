import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import PreTrainedModel, AutoModel
from transformers.modeling_outputs import TokenClassifierOutput, SequenceClassifierOutput
from torchcrf import CRF
from torch.nn.utils.rnn import pad_sequence


class ModelForJointClassification(PreTrainedModel):
    authorized_unexpected_keys = [r"pooler"]
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, model_name, config):
        super().__init__(config)

        self.num_token_labels = config.num_token_labels
        self.num_labels = config.num_labels
        self.token_label_map = config.token_label_map
        self.model = AutoModel.from_pretrained(model_name, config = config, add_pooling_layer=False)
        self.seq_classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.token_classifier = nn.Linear(config.hidden_size, self.num_token_labels)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            seq_labels=None,
            token_labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            token_class_weight=None,
            seq_class_weight=None,
            token_lambda=1,
            **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]

        token_sequence_output = self.dropout(sequence_output)
        token_logits = self.token_classifier(token_sequence_output)
        # da shape [32, 128, 2] a shape [32, 2] lo fa implementazione ufficiale 
        sequence_output = sequence_output[:, 0, :]
        seq_logits = self.seq_classifier(sequence_output)

        loss = None
        if token_labels is not None:
            token_loss_fct = CrossEntropyLoss(weight=token_class_weight)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = token_logits.view(-1, self.num_token_labels)
                active_labels = torch.where(
                    active_loss, token_labels.view(-1), torch.tensor(token_loss_fct.ignore_index).type_as(token_labels)
                )
                token_loss = token_loss_fct(active_logits, active_labels)
            else:
                token_loss = token_loss_fct(token_logits.view(-1, self.num_token_labels), token_labels.view(-1))
            loss = token_loss
        if seq_labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                seq_loss_fct = MSELoss()
                seq_loss = seq_loss_fct(seq_logits.view(-1), seq_labels.view(-1))
            else:
                seq_loss_fct = CrossEntropyLoss(weight=seq_class_weight)
                seq_loss = seq_loss_fct(seq_logits.view(-1, self.num_labels), seq_labels.view(-1))
            loss = token_lambda * loss + seq_loss if loss is not None else seq_loss

        output = (token_logits, seq_logits) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class ModelForJointClassificationDeberta(PreTrainedModel):
    authorized_unexpected_keys = [r"pooler"]
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, model_name, config):
        super().__init__(config)

        self.num_token_labels = config.num_token_labels
        self.num_labels = config.num_labels
        self.token_label_map = config.token_label_map
        self.model = AutoModel.from_pretrained(model_name, config = config)
        self.seq_classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.token_classifier = nn.Linear(config.hidden_size, self.num_token_labels)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            seq_labels=None,
            token_labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            token_class_weight=None,
            seq_class_weight=None,
            token_lambda=1,
            **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]

        token_sequence_output = self.dropout(sequence_output)
        token_logits = self.token_classifier(token_sequence_output)
        # da shape [32, 128, 2] a shape [32, 2] lo fa implementazione ufficiale 
        sequence_output = sequence_output[:, 0, :]
        seq_logits = self.seq_classifier(sequence_output)

        loss = None
        if token_labels is not None:
            token_loss_fct = CrossEntropyLoss(weight=token_class_weight)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = token_logits.view(-1, self.num_token_labels)
                active_labels = torch.where(
                    active_loss, token_labels.view(-1), torch.tensor(token_loss_fct.ignore_index).type_as(token_labels)
                )
                token_loss = token_loss_fct(active_logits, active_labels)
            else:
                token_loss = token_loss_fct(token_logits.view(-1, self.num_token_labels), token_labels.view(-1))
            loss = token_loss
        if seq_labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                seq_loss_fct = MSELoss()
                seq_loss = seq_loss_fct(seq_logits.view(-1), seq_labels.view(-1))
            else:
                seq_loss_fct = CrossEntropyLoss(weight=seq_class_weight)
                seq_loss = seq_loss_fct(seq_logits.view(-1, self.num_labels), seq_labels.view(-1))
            loss = token_lambda * loss + seq_loss if loss is not None else seq_loss

        output = (token_logits, seq_logits) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class ModelForJointClassificationWithCRF(PreTrainedModel):
    authorized_unexpected_keys = [r"pooler"]
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, model_name, config):
        super().__init__(config)
        self.num_token_labels = config.num_token_labels
        self.num_labels = config.num_labels
        self.token_label_map = config.token_label_map
        self.model = AutoModel.from_pretrained(model_name, config = config, add_pooling_layer=False)
        self.seq_classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.token_classifier = nn.Linear(config.hidden_size, self.num_token_labels)
        self.crf = CRF(num_tags=self.num_token_labels, batch_first=True)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            seq_labels=None,
            token_labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            token_class_weight=None,
            seq_class_weight=None,
            token_lambda=1,
            **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]

        token_sequence_output = self.dropout(sequence_output)
        token_logits = self.token_classifier(token_sequence_output)
        # da shape [32, 128, 2] a shape [32, 2] lo fa implementazione ufficiale 
        sequence_output = sequence_output[:, 0, :]
        seq_logits = self.seq_classifier(sequence_output)

        loss = None
        if token_labels is not None:

            new_token_logits_list, new_token_labels_list = [], []
            for t_logits, t_labels in zip(token_logits, token_labels):
                # Index logits and labels using prediction mask to pass only the
                # first subtoken of each word to CRF.
                new_token_logits_list.append(t_logits[t_labels >= 0])
                new_token_labels_list.append(t_labels[t_labels >= 0])

            new_token_logits = pad_sequence(new_token_logits_list).transpose(0, 1)
            new_token_labels = pad_sequence(new_token_labels_list, padding_value=-999).transpose(0, 1)
            token_prediction_mask = new_token_labels >= 0
            active_token_labels = torch.where(
                token_prediction_mask, new_token_labels, torch.tensor(0).type_as(new_token_labels)
            )

            loss = -torch.mean(
                self.crf(new_token_logits, active_token_labels, token_prediction_mask, reduction='token_mean'))
            output_tags = self.crf.decode(new_token_logits, token_prediction_mask)

        if seq_labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                seq_loss_fct = MSELoss()
                seq_loss = seq_loss_fct(seq_logits.view(-1), seq_labels.view(-1))
            else:
                seq_loss_fct = CrossEntropyLoss(weight=seq_class_weight)
                seq_loss = seq_loss_fct(seq_logits.view(-1, self.num_labels), seq_labels.view(-1))
            loss = token_lambda * loss + seq_loss if loss is not None else seq_loss

        if token_labels is not None:
            output = (token_logits, seq_logits, output_tags) + outputs[2:]
        else:
            output = (token_logits, seq_logits) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class ModelForJointClassificationWithCRFDeberta(PreTrainedModel):
    authorized_unexpected_keys = [r"pooler"]
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, model_name, config):
        super().__init__(config)
        self.num_token_labels = config.num_token_labels
        self.num_labels = config.num_labels
        self.token_label_map = config.token_label_map
        self.model = AutoModel.from_pretrained(model_name, config = config)
        self.seq_classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.token_classifier = nn.Linear(config.hidden_size, self.num_token_labels)
        self.crf = CRF(num_tags=self.num_token_labels, batch_first=True)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            seq_labels=None,
            token_labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            token_class_weight=None,
            seq_class_weight=None,
            token_lambda=1,
            **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]

        token_sequence_output = self.dropout(sequence_output)
        token_logits = self.token_classifier(token_sequence_output)
        # da shape [32, 128, 2] a shape [32, 2] lo fa implementazione ufficiale 
        sequence_output = sequence_output[:, 0, :]
        seq_logits = self.seq_classifier(sequence_output)

        loss = None
        if token_labels is not None:

            new_token_logits_list, new_token_labels_list = [], []
            for t_logits, t_labels in zip(token_logits, token_labels):
                # Index logits and labels using prediction mask to pass only the
                # first subtoken of each word to CRF.
                new_token_logits_list.append(t_logits[t_labels >= 0])
                new_token_labels_list.append(t_labels[t_labels >= 0])

            new_token_logits = pad_sequence(new_token_logits_list).transpose(0, 1)
            new_token_labels = pad_sequence(new_token_labels_list, padding_value=-999).transpose(0, 1)
            token_prediction_mask = new_token_labels >= 0
            active_token_labels = torch.where(
                token_prediction_mask, new_token_labels, torch.tensor(0).type_as(new_token_labels)
            )

            loss = -torch.mean(
                self.crf(new_token_logits, active_token_labels, token_prediction_mask, reduction='token_mean'))
            output_tags = self.crf.decode(new_token_logits, token_prediction_mask)

        if seq_labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                seq_loss_fct = MSELoss()
                seq_loss = seq_loss_fct(seq_logits.view(-1), seq_labels.view(-1))
            else:
                seq_loss_fct = CrossEntropyLoss(weight=seq_class_weight)
                seq_loss = seq_loss_fct(seq_logits.view(-1, self.num_labels), seq_labels.view(-1))
            loss = token_lambda * loss + seq_loss if loss is not None else seq_loss

        if token_labels is not None:
            output = (token_logits, seq_logits, output_tags) + outputs[2:]
        else:
            output = (token_logits, seq_logits) + outputs[2:]
        return ((loss,) + output) if loss is not None else output