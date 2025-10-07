"""Utils for preparing datasets before using litcode. 
Adapted from Huth ridge utils. 
TODO: Add proper citation"""
from typing import List, Tuple, Union
from pathlib import Path
import json
import numpy as np
from .textgrid import TextGrid
import pickle
import h5py

DEFAULT_BAD_WORDS = frozenset(["sentence_start", "sentence_end", "br", "lg", "ls", "ns", "sp"])




##########Transcript Preprocessing Utils###########

def create_lebel_transcripts( story_list: List[str], 
                       textgrids_dir: Union[str, Path],
                       respdict_path: Union[str, Path],
                       output_dir: Union[str, Path],
                       file_name: str = "lebel_transcripts.pkl"
                       ) -> None:

    """Create transcripts for the given stories and save them to the output directory.

    Args:
        story_list: List of story names to process
        textgrids_dir: Directory containing the TextGrid files
        respdict_path: Path to the response dictionary JSON file
        output_dir: Directory to save the generated transcripts
    """

    text_grids = _load_textgrids(story_list, textgrids_dir)
    with open(respdict_path, "r") as f:
        respdict = json.load(f)
    tr_times = _simulate_trtimes(story_list, respdict)
    processed_transcripts = _process_textgrids(text_grids, tr_times)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / file_name
    with open(output_path, "wb") as f:
        pickle.dump(processed_transcripts, f)
    

def _load_textgrids(stories: List[str], textgrids_dir: Union[str, Path]) -> dict:
    """Load TextGrid files for the given stories from the specified TextGrid directory.

    Args:
        stories: List of story names
        data_dir: Directory containing the TextGrid files

    Returns:
        Dictionary mapping story names to their corresponding TextGrid objects
    """

    grids = {}
    for story in stories:
        grid_path = Path(textgrids_dir) / f"{story}.TextGrid"
        grids[story] = TextGrid.load(grid_path)
    return grids
    

def _simulate_trtimes(stories: List[str], respdict: dict, tr: float = 2.0, start_time: float = 10.0, pad: int = 10) -> dict:
    """Simulate TR times for the given stories based on the response dictionary.

    Args:
        stories: List of story names
        respdict: Dictionary mapping story names to their response lengths
        tr: Expected TR value
        start_time: Start time for the simulation
        pad: Padding to subtract from the response length

    Returns:
        Dictionary mapping story names to their simulated TR times
    """
    tr_times = {}
    for story in stories:
        resp_length = respdict.get(story, 0)
        tr_times[story] = list(np.arange(-start_time, (resp_length - pad) * tr, tr))
    return tr_times

def _process_textgrids(text_grids: dict, 
                       tr_times: dict, 
                       bad_words: frozenset = DEFAULT_BAD_WORDS
                    ) -> dict[dict]:
    """Process the loaded TextGrid files to extract word sequences, filtering out bad words.

    Args:
        text_grids: Dictionary mapping story names to their corresponding TextGrid objects
        bad_words: Set of words to filter out from the transcripts
    """
    processed_transcripts = {}
    for story in text_grids.keys():
        simple_transcript = text_grids[story].tiers[1].make_simple_transcript()
        ## Filter out bad words
        filtered_transcript = [x for x in simple_transcript if x[2].lower().strip("{}").strip() not in bad_words]
        # Further processing can be done here as needed
        processed_transcripts[story] = _process_single_story(filtered_transcript, tr_times[story])

    return processed_transcripts

def _process_single_story(processed_transcript: List[Tuple], 
                          tr_times: List[float]) -> dict:
    """Process a single story's transcript and TR times to create a structured representation.
    Args:
        proceesed_transcript: List of tuples representing the transcript (start_time, end_time, word)
        tr_times: List of TR times for the story
    Returns:   
        Tuple containing processed story information
    """
    
    data_entries = list(zip(*processed_transcript))[2]
    if isinstance(data_entries[0], str):
        data = list(map(str.lower, list(zip(*processed_transcript))[2]))
    else:
        data = data_entries
    word_starts = np.array(list(map(float, list(zip(*processed_transcript))[0])))
    word_ends = np.array(list(map(float, list(zip(*processed_transcript))[1])))
    word_avgtimes = (word_starts + word_ends)/2.0
    
    tr = np.mean(np.diff(tr_times))
    tr_midpoints = np.array(tr_times) + tr/2.0

    split_inds = [(word_starts<(t+tr)).sum() for t in tr_times][:-1]
    return {"words": data, "split_indices": split_inds, "data_times":word_avgtimes,"tr_times": tr_midpoints}

def create_brain_response_dict(story_list: List[str], 
                       resp_data_dir: Union[str, Path],
                       output_dir: Union[str, Path],
                       file_name: str = "brain_resp_huge.pkl"
                       ) -> None:
    """Create a dictionary of brain responses for the given stories and save it to the output path.

    Args:
        story_list: List of story names to process
        neural_data_dir: Directory containing the neural data files
        output_path: Path to save the generated brain response dictionary
        file_name: Name of the output file
    """

    brain_responses = {}
    for story in story_list:
        resp_data_path = Path(resp_data_dir) / f"{story}.hf5"
        with h5py.File(resp_data_path, "r") as f:
            brain_responses[story] = f["data"][:]

    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with open(output_dir / file_name, "wb") as f:
        pickle.dump(brain_responses, f)