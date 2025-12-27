"""
Luframe Transcription Agent

A LiveKit agent that provides real-time transcription for all meeting participants.
The agent joins the room invisibly and transcribes all audio, publishing transcriptions
via LiveKit's text streams to the frontend.

This agent uses multi-track transcription to handle all participants simultaneously.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import stt, JobContext, WorkerOptions, cli, AutoSubscribe
from livekit.plugins import deepgram

from transcription_config import get_stt_language, get_stt_model

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("luframe-transcription")


class ParticipantTranscriber:
    """Handles transcription for a single participant's audio track."""

    def __init__(
        self,
        room: rtc.Room,
        participant: rtc.RemoteParticipant,
        track: rtc.RemoteAudioTrack,
        stt_instance: stt.STT,
    ):
        self.room = room
        self.participant = participant
        self.track = track
        self.stt = stt_instance
        self._task: asyncio.Task | None = None
        self._segment_counter = 0  # Counter for unique segment IDs

    async def start(self):
        """Start transcribing this participant's audio."""
        self._task = asyncio.create_task(self._transcribe_track())

    async def stop(self):
        """Stop transcribing."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _transcribe_track(self):
        """Process audio from the track and publish transcriptions."""
        try:
            audio_stream = rtc.AudioStream(self.track)

            # Create a streaming STT session
            stt_stream = self.stt.stream()

            async def process_audio():
                async for audio_event in audio_stream:
                    stt_stream.push_frame(audio_event.frame)
                # Signal end of input so final transcripts flush when tracks stop
                stt_stream.end_input()

            async def process_transcriptions():
                current_segment_id = None
                async for event in stt_stream:
                    if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                        transcript_text = event.alternatives[0].text if event.alternatives else ""
                        if transcript_text.strip():
                            # Use the current segment ID or create a new one
                            if current_segment_id is None:
                                self._segment_counter += 1
                                current_segment_id = f"{self.participant.identity}-{self._segment_counter}"

                            logger.info(
                                f"[{self.participant.name or self.participant.identity}] {transcript_text}"
                            )
                            # Publish transcription via text stream
                            await self._publish_transcription(
                                transcript_text,
                                is_final=True,
                                segment_id=current_segment_id,
                            )
                            # Reset for next segment
                            current_segment_id = None
                    elif event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
                        transcript_text = event.alternatives[0].text if event.alternatives else ""
                        if transcript_text.strip():
                            # Create segment ID if this is a new utterance
                            if current_segment_id is None:
                                self._segment_counter += 1
                                current_segment_id = f"{self.participant.identity}-{self._segment_counter}"

                            await self._publish_transcription(
                                transcript_text,
                                is_final=False,
                                segment_id=current_segment_id,
                            )

            # Run both tasks concurrently
            await asyncio.gather(process_audio(), process_transcriptions())

        except asyncio.CancelledError:
            logger.info(f"Stopped transcribing {self.participant.identity}")
            raise
        except Exception as e:
            logger.error(f"Error transcribing {self.participant.identity}: {e}")

    async def _publish_transcription(self, text: str, is_final: bool, segment_id: str):
        """Publish transcription to all participants via text stream."""
        try:
            # Use the lk.transcription topic that the frontend expects
            await self.room.local_participant.send_text(
                text,
                topic="lk.transcription",
                attributes={
                    "lk.transcribed_track_id": self.track.sid,
                    "lk.transcription_final": str(is_final).lower(),
                    "lk.segment_id": segment_id,
                    "speaker_identity": self.participant.identity,
                    "speaker_name": self.participant.name or self.participant.identity,
                },
            )
        except Exception as e:
            logger.error(f"Error publishing transcription: {e}")


class MultiParticipantTranscriber:
    """Manages transcription for all participants in a room."""

    def __init__(self, room: rtc.Room):
        self.room = room
        self.transcribers: Dict[str, ParticipantTranscriber] = {}
        self.stt = deepgram.STT(
            model=get_stt_model(),
            language=get_stt_language(),
        )

    async def start(self):
        """Start listening for participants and their audio tracks."""
        # Set up event handlers
        self.room.on("track_subscribed", self._on_track_subscribed)
        self.room.on("track_unsubscribed", self._on_track_unsubscribed)
        self.room.on("track_published", self._on_track_published)
        self.room.on("participant_connected", self._on_participant_connected)
        self.room.on("participant_disconnected", self._on_participant_disconnected)

        logger.info(f"Found {len(self.room.remote_participants)} remote participants")

        # Subscribe to existing participants' audio tracks
        for participant in self.room.remote_participants.values():
            logger.info(f"Checking participant: {participant.identity}, tracks: {len(participant.track_publications)}")
            for track_pub in participant.track_publications.values():
                logger.info(f"  Track: {track_pub.sid}, kind: {track_pub.kind}, subscribed: {track_pub.subscribed}")
                if (
                    track_pub.track
                    and track_pub.kind == rtc.TrackKind.KIND_AUDIO
                    and isinstance(track_pub.track, rtc.RemoteAudioTrack)
                ):
                    await self._start_transcriber(participant, track_pub.track)

    async def stop(self):
        """Stop all transcribers."""
        for transcriber in self.transcribers.values():
            await transcriber.stop()
        self.transcribers.clear()

    def _on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        """Handle new audio track subscription."""
        logger.info(f"Track subscribed: {track.sid}, kind: {track.kind}, from: {participant.identity}")
        if track.kind == rtc.TrackKind.KIND_AUDIO and isinstance(
            track, rtc.RemoteAudioTrack
        ):
            logger.info(f"Starting transcription for audio track from {participant.identity}")
            asyncio.create_task(self._start_transcriber(participant, track))

    def _on_track_unsubscribed(
        self,
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        """Handle audio track unsubscription."""
        key = f"{participant.identity}:{track.sid}"
        if key in self.transcribers:
            asyncio.create_task(self.transcribers[key].stop())
            del self.transcribers[key]

    def _on_track_published(
        self,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        """Handle when a track is published (before subscription)."""
        logger.info(f"Track published: {publication.sid}, kind: {publication.kind}, from: {participant.identity}")

    def _on_participant_connected(self, participant: rtc.RemoteParticipant):
        """Handle when a new participant joins."""
        logger.info(f"Participant connected: {participant.identity}, name: {participant.name}")

    def _on_participant_disconnected(self, participant: rtc.RemoteParticipant):
        """Handle participant disconnection."""
        keys_to_remove = [
            k for k in self.transcribers if k.startswith(f"{participant.identity}:")
        ]
        for key in keys_to_remove:
            asyncio.create_task(self.transcribers[key].stop())
            del self.transcribers[key]

    async def _start_transcriber(
        self, participant: rtc.RemoteParticipant, track: rtc.RemoteAudioTrack
    ):
        """Start a transcriber for a participant's audio track."""
        key = f"{participant.identity}:{track.sid}"
        if key in self.transcribers:
            return  # Already transcribing this track

        logger.info(
            f"Starting transcription for {participant.name or participant.identity}"
        )
        transcriber = ParticipantTranscriber(
            self.room, participant, track, self.stt
        )
        self.transcribers[key] = transcriber
        await transcriber.start()


async def entrypoint(ctx: JobContext):
    """
    Main entrypoint for the transcription agent.

    This agent:
    1. Joins the LiveKit room invisibly (as a hidden agent)
    2. Subscribes to all participant audio tracks
    3. Runs STT (Speech-to-Text) on all audio
    4. Publishes transcriptions via LiveKit text streams (lk.transcription topic)
    """
    logger.info(f"Transcription agent starting for room: {ctx.room.name}")

    # Connect to the room with audio subscription
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    logger.info("Connected to room, starting multi-participant transcriber")

    # Create and start the multi-participant transcriber
    transcriber = MultiParticipantTranscriber(ctx.room)
    await transcriber.start()

    logger.info("Transcription agent is now listening to all participants")

    # Keep the agent running
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        logger.info("Agent shutting down")
        await transcriber.stop()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name="transcription-agent",  # Must match the name in token dispatch
    ))
