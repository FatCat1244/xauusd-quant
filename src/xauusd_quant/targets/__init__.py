"""The target factory (Prompt #8): outcomes, stored apart from every feature.

Every column is prefixed ``target_`` and uses bars after ``t``; the feature
factory refuses such columns, and :mod:`.alignment` checks that a feature at
``t`` meets exactly the outcome over ``t+1 .. t+h``.
"""
