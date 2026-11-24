---------------------------- MODULE Incarnation ----------------------------
EXTENDS Naturals, FiniteSets

(*
 * Bounded incarnation-epoch and stale-reply safety protocol for Coordinator/Session.
 * Enforces single_active_incarnation and stale_reply_rejection across respawns
 * and out-of-order/replayed participant replies.
 *)

CONSTANTS Participants, TxnIds, MaxIncarnation

VARIABLES
    active_inc,      \* Current coordinator incarnation in 1..MaxIncarnation
    active_txn,      \* Active transaction id (0 when idle)
    live_incs,       \* Subset of 1..MaxIncarnation currently permitted to commit
    in_flight,       \* Set of reply records [p |-> p, inc |-> i, txn |-> t]
    accepted_replies \* Set of [p, inc, txn, accepted_at_inc, accepted_at_txn]

vars == <<active_inc, active_txn, live_incs, in_flight, accepted_replies>>

TypeOK ==
    /\ active_inc \in 1..MaxIncarnation
    /\ active_txn \in ({0} \cup TxnIds)
    /\ live_incs \subseteq (1..MaxIncarnation)

Init ==
    /\ active_inc = 1
    /\ active_txn = 0
    /\ live_incs = {1}
    /\ in_flight = {}
    /\ accepted_replies = {}

StartTxn(t) ==
    /\ active_txn = 0
    /\ t \in TxnIds
    /\ active_txn' = t
    /\ UNCHANGED <<active_inc, live_incs, in_flight, accepted_replies>>

SendReply(p) ==
    /\ active_txn # 0
    /\ in_flight' = in_flight \cup {[p |-> p, inc |-> active_inc, txn |-> active_txn]}
    /\ UNCHANGED <<active_inc, active_txn, live_incs, accepted_replies>>

DeliverFreshReply(r) ==
    /\ r \in in_flight
    /\ r.inc = active_inc
    /\ r.txn = active_txn
    /\ accepted_replies' = accepted_replies \cup
           {[p |-> r.p, inc |-> r.inc, txn |-> r.txn,
             accepted_at_inc |-> active_inc, accepted_at_txn |-> active_txn]}
    /\ in_flight' = in_flight \ {r}
    /\ UNCHANGED <<active_inc, active_txn, live_incs>>

RejectStaleReply(r) ==
    /\ r \in in_flight
    /\ (r.inc # active_inc \/ r.txn # active_txn)
    /\ in_flight' = in_flight \ {r}
    /\ UNCHANGED <<active_inc, active_txn, live_incs, accepted_replies>>

FinishTxn ==
    /\ active_txn # 0
    /\ active_txn' = 0
    /\ UNCHANGED <<active_inc, live_incs, in_flight, accepted_replies>>

Respawn ==
    /\ active_inc < MaxIncarnation
    /\ active_inc' = active_inc + 1
    /\ active_txn' = 0
    /\ live_incs' = {active_inc + 1}
    /\ UNCHANGED <<in_flight, accepted_replies>>

Next ==
    \/ (\E t \in TxnIds : StartTxn(t))
    \/ (\E p \in Participants : SendReply(p))
    \/ (\E r \in in_flight : DeliverFreshReply(r))
    \/ (\E r \in in_flight : RejectStaleReply(r))
    \/ FinishTxn
    \/ Respawn

\* Safety invariants
SingleActiveIncarnation ==
    /\ Cardinality(live_incs) = 1
    /\ live_incs = {active_inc}

StaleReplyRejection ==
    \A r \in accepted_replies :
        r.inc = r.accepted_at_inc /\ r.txn = r.accepted_at_txn

Spec == Init /\ [][Next]_vars
=============================================================================
