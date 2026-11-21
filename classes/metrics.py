import tensorflow as tf

class FBetaScore(tf.keras.metrics.Metric):
    """
    Computes the F-beta score.
    """
    def __init__(self, beta=1.0, name='fbeta_score', **kwargs):
        super(FBetaScore, self).__init__(name=name, **kwargs)
        self.beta = beta
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        if len(y_pred.shape) > 1 and y_pred.shape[-1] > 1:
            y_true = tf.argmax(y_true, axis=-1)
            y_pred = tf.argmax(y_pred, axis=-1)

        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(tf.round(y_pred), tf.bool)

        true_pos = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        false_pos = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))
        false_neg = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False))

        # We assume sample_weight is None for simplicity here, but can be scaled
        self.true_positives.assign_add(tf.reduce_sum(tf.cast(true_pos, self.dtype)))
        self.false_positives.assign_add(tf.reduce_sum(tf.cast(false_pos, self.dtype)))
        self.false_negatives.assign_add(tf.reduce_sum(tf.cast(false_neg, self.dtype)))

    def result(self):
        p = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        r = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
        beta_sq = self.beta ** 2
        return (1 + beta_sq) * (p * r) / ((beta_sq * p) + r + tf.keras.backend.epsilon())

    def reset_state(self):
        self.true_positives.assign(0.0)
        self.false_positives.assign(0.0)
        self.false_negatives.assign(0.0)

class F1Score(FBetaScore):
    """
    Computes the F1-score (F-beta with beta=1.0)
    """
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(beta=1.0, name=name, **kwargs)

class NPV(tf.keras.metrics.Metric):
    """
    Computes Negative Predictive Value (NPV). 
    NPV = TN / (TN + FN)
    """
    def __init__(self, name='npv', **kwargs):
        super(NPV, self).__init__(name=name, **kwargs)
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        if len(y_pred.shape) > 1 and y_pred.shape[-1] > 1:
            y_true = tf.argmax(y_true, axis=-1)
            y_pred = tf.argmax(y_pred, axis=-1)
            
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(tf.round(y_pred), tf.bool)

        true_neg = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, False))
        false_neg = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False))

        self.true_negatives.assign_add(tf.reduce_sum(tf.cast(true_neg, self.dtype)))
        self.false_negatives.assign_add(tf.reduce_sum(tf.cast(false_neg, self.dtype)))

    def result(self):
        return self.true_negatives / (self.true_negatives + self.false_negatives + tf.keras.backend.epsilon())

    def reset_state(self):
        self.true_negatives.assign(0.0)
        self.false_negatives.assign(0.0)
