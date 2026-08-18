class BrokerError(Exception):
    retryable = False


class TransientBrokerError(BrokerError):
    retryable = True


class PermanentBrokerError(BrokerError):
    retryable = False