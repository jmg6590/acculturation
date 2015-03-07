
import os
import sys
import cPickle
import glob
from collections import defaultdict

from acculturation.lingdistance.jensen_shannon import jensen_shannon_distances
import corpcorp


"""
Contains functions to easily compute JS distances over
sets of corpcorp Members with flexible segmentation

See __main__ section at bottom of file for sample experiments
"""


################################################
# Member-level distances functions
# The flexible workhorse functions are get_distances and get_monthly_distances
# get_userlevel_distances and get_dyadic_distances are wrappers


def get_userlevel_distances(input_dir, monthly=False, **get_distances_kargs):
    """
    Wrapper around get_distances. For get_distances_kargs, see get_distances documentation.

    This fnc just sets a default for get_member_message_segmentation_fnc
    and chooses between get_distances and get_monthly_distances for you
    """
    if monthly:
        distances = get_monthly_distances(input_dir, 
                        get_member_message_segmentation_fnc=get_userlevel_segmentation_fnc, 
                        **get_distances_kargs)
    else:
        distances = get_distances(input_dir, 
                        get_member_message_segmentation_fnc=get_userlevel_segmentation_fnc, 
                        **get_distances_kargs)
    return distances




def get_dyadic_distances(input_dir, monthly=False, **get_distances_kargs):
    """
    Wrapper around get_distances. For get_distances_kargs, see get_distances documentation.

    This fnc just sets a default for get_member_message_segmentation_fnc
    and chooses between get_distances and get_monthly_distances for you
    """
    if monthly:
        distances = get_monthly_distances(input_dir, 
                        get_member_message_segmentation_fnc=get_dyadic_segmentation_fnc, 
                        **get_distances_kargs)
    else:
        distances = get_distances(input_dir, 
                        get_member_message_segmentation_fnc=get_dyadic_segmentation_fnc, 
                        **get_distances_kargs)
    return distances




def get_distances(input_dir, get_member_message_segmentation_fnc=None,
                                liwc_map=False, vocabsize=1000,
                                sampling=False, sampsize=1000,
                                min_segment_size=5):
    """
    Measures JS distances for an input_dir of corpcorp.Member json files

    args:
        input_dir- where all the *.json email files live

    kargs:
        get_member_message_segmentation_fnc- 
            Controls which messages we compare to compute distances on a member level.
            Takes a corpcorp.Member object, returns a msg-level segementation callback
            The JS code in lingdistance.jensen_shannon exposes segmentation control 
                via a callback. Since we may require a different segmentation callback
                for each Member (e.g., if the segmentation depends on the Member's username),
                this function in turn supports an additional callback to return 
                a new segementation callback on a member level.
            If no callback is provided, we by default segment on a user level
                (i.e., between a user and his/her interlocuters)
            See segmentation callback code below for samples
            Note: we assume that member.username will be one of the segments for each member!

        other kargs are for the JS distance computation. 
        see acculturation.lingdistance.jensen_shannon for documentation

    return value:
        distances dict, keyed by member username
            {username: {segment: JS distance float}}
    """

    if not get_member_message_segmentation_fnc:
        # Default to user and interlocuters
        get_member_message_segmentation_fnc = get_userlevel_segmentation_fnc

    distances = {}
    filenames = glob.glob(os.path.join(input_dir, "*.json"))
    for filename in filenames:

        member = corpcorp.Member(filename)
        msg_segmentation_fnc = get_member_message_segmentation_fnc(member)

        # msg_segmentation_fnc will determine how msgs get put into buckets.
        member_distances = jensen_shannon_distances(member.messages, 
                                doc_segmentation_fnc=msg_segmentation_fnc,
                                liwc_map=liwc_map, vocabsize=vocabsize,
                                sampling=sampling, sampsize=sampsize,
                                min_segment_size=min_segment_size)

        # Add to distances
        distances[member.username] = {}
        for a,b in member_distances:
            # Want whatever this member was getting compared to
            # Note: member.username must have been a possible segment
            #   returned by get_member_message_segmentation_fnc!
            seg = a if member.username not in a else b
            distances[member.username][seg] = member_distances[(a,b)]

    return distances



def get_monthly_distances(input_dir, get_member_message_segmentation_fnc=None,
                                liwc_map=False, vocabsize=1000,
                                sampling=False, sampsize=1000,
                                min_segment_size=5):
    """
    Behavior and args similar to get_distances, distance computation just broken up 
        to occur on a monthly basis
    Return dict same as get_distances, 
        member-level distances just broken up by month:
            {user: {(YYYY, MM): {segment: d}}}
    """
    monthly_distances = {}

    if not get_member_message_segmentation_fnc:
        # Default to user and interlocuters
        get_member_message_segmentation_fnc = get_userlevel_segmentation_fnc

    filenames = glob.glob(os.path.join(input_dir, "*.json"))
    for filename in filenames:

        member = corpcorp.Member(filename)
        msg_segmentation_fnc = get_member_message_segmentation_fnc(member)
        monthly_distances[member.username] = {}

        # Break up messages by month
        months2msgs = defaultdict(list)
        for msg in member.messages:
            date = (msg.date.year, msg.date.month)
            months2msgs[date].append(msg)

        # Compute JS by month
        for month, msgs in months2msgs.iteritems():
            distances = jensen_shannon_distances(msgs, 
                                doc_segmentation_fnc=msg_segmentation_fnc,
                                liwc_map=liwc_map, vocabsize=vocabsize,
                                sampling=sampling, sampsize=sampsize,
                                min_segment_size=min_segment_size)
            
            # Add to monthly_distances
            monthly_distances[member.username][month] = {}
            for a,b in distances:
                seg = a if member.username not in a else b
                monthly_distances[member.username][month][seg] = distances[(a,b)]       

    return monthly_distances


################################################
# Segmentation callbacks
# These fncs generate the callbacks
# we use to segment messages. 
# When we generate JS distances,
# we will compare these segments.


def get_dyadic_segmentation_fnc(member):
    """ compare each user to each other user """
    def dyadic_segmentation_fnc(msg):
        # Just care who this message was from
        # (but msg.frm might be messy, from email header)
        if member.username in msg.frm:
            # Msg from our src_user
            return [member.username]
        else:
            # An interlocuter
            return [msg.frm]
    return dyadic_segmentation_fnc


def get_userlevel_segmentation_fnc(member):
    """ compare each user to set of all of user's interlocuters """
    def user_level_segmentation_fnc(msg):
        if member.username in msg.frm:
            # Msg from our src_user
            return [member.username]
        else:
            # An interlocuter
            return ["other"]
    return user_level_segmentation_fnc




if __name__ == "__main__":

    input_dir = "dummydata"


    #################
    # User-level experiment-- distance between a member and his/her interlocuters

    print "\nUser-level experiment"

    segmentation_fnc_callback = get_userlevel_segmentation_fnc

    # With LIWC:
    distances = get_distances(input_dir, get_member_message_segmentation_fnc=segmentation_fnc_callback,
                                liwc_map=True, vocabsize=1000,
                                sampling=False, sampsize=1000,
                                min_segment_size=5)
    print "\n\nWith LIWC: ", distances

    # Without LIWC:
    distances = get_distances(input_dir, get_member_message_segmentation_fnc=segmentation_fnc_callback,
                                liwc_map=False, vocabsize=1000,
                                sampling=False, sampsize=1000,
                                min_segment_size=5)
    print "\n\nWithout LIWC: ", distances

    # Monthly
    distances = get_monthly_distances(input_dir, get_member_message_segmentation_fnc=segmentation_fnc_callback,
                                liwc_map=False, vocabsize=1000,
                                sampling=False, sampsize=1000,
                                min_segment_size=5)
    print "\n\nMonthly: ", distances
    


    #################
    # Dyadic experiment-- distance between a member and his/her interlocuters

    print "\nDyadic experiment"
    
    segmentation_fnc_callback = get_dyadic_segmentation_fnc

    distances = get_distances(input_dir, get_member_message_segmentation_fnc=segmentation_fnc_callback,
                                liwc_map=True, vocabsize=1000,
                                sampling=False, sampsize=1000,
                                min_segment_size=5)
    print "\n", distances

    print "\n\n"




