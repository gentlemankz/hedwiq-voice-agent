"""
Participant transcription worker extracted from luframe_agent.
Keeps audio→text flow modular and reusable.

Phase 1 (Real-Time Actions) Addition:
- Added action_classifier parameter to feed transcript context for classification

Phase 3 (Real-Time Actions) Addition:
- Added email_draft_generator parameter to feed transcript context for draft generation
"""

import asyncio
import logging
import time
from typing import Optional

from livekit import rtc
from livekit.agents import stt

from insight_analyzer import TranscriptEntry, InsightAnalyzer
from document_referencer import DocumentReferencer

# Type hint import for agenda tracker, action classifier, and email draft generator (avoids circular import)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agenda_tracker import AgendaTracker
    from action_classifier import ActionClassifier
    from email_draft_generator import EmailDraftGenerator

logger = logging.getLogger("luframe-agent")


class ParticipantTranscriber:
    """Handles transcription for a single participant's audio track."""

    def __init__(
        self,
        room: rtc.Room,
        participant: rtc.RemoteParticipant,
        track: rtc.RemoteAudioTrack,
        stt_instance: stt.STT,
        insight_analyzer: InsightAnalyzer,
        document_referencer: Optional[DocumentReferencer] = None,
        agenda_tracker: Optional["AgendaTracker"] = None,
        action_classifier: Optional["ActionClassifier"] = None,
        email_draft_generator: Optional["EmailDraftGenerator"] = None,
        transcription_topic: str = "lk.transcription",
    ):
        self.room = room
        self.participant = participant
        self.track = track
        self.stt = stt_instance
        self.insight_analyzer = insight_analyzer
        self.document_referencer = document_referencer
        self.agenda_tracker = agenda_tracker
        self.action_classifier = action_classifier
        self.email_draft_generator = email_draft_generator
        self.transcription_topic = transcription_topic
        self._task: Optional[asyncio.Task] = None
        self._segment_counter = 0
        self._segment_start_time: Optional[float] = None

    async def start(self):
        self._task = asyncio.create_task(self._transcribe_track())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _transcribe_track(self):
        try:
            audio_stream = rtc.AudioStream(self.track)
            stt_stream = self.stt.stream()

            async def process_audio():
                async for audio_event in audio_stream:
                    stt_stream.push_frame(audio_event.frame)
                stt_stream.end_input()

            async def process_transcriptions():
                current_segment_id = None
                async for event in stt_stream:
                    if event.type == stt.SpeechEventType.START_OF_SPEECH:
                        self._segment_counter += 1
                        current_segment_id = (
                            f"{self.participant.identity}-{self._segment_counter}"
                        )
                        self._segment_start_time = time.time()

                    elif event.type == stt.SpeechEventType.END_OF_SPEECH:
                        logger.debug(
                            f"Speech ended for {self.participant.identity}, segment: {current_segment_id}"
                        )

                    elif event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                        transcript_text = (
                            event.alternatives[0].text if event.alternatives else ""
                        )
                        if transcript_text.strip():
                            if current_segment_id is None:
                                self._segment_counter += 1
                                current_segment_id = (
                                    f"{self.participant.identity}-{self._segment_counter}"
                                )

                            duration_seconds = 2.0
                            if self._segment_start_time:
                                duration_seconds = time.time() - self._segment_start_time

                            logger.info(
                                f"[{self.participant.name or self.participant.identity}] {transcript_text}"
                            )

                            await self._publish_transcription(
                                transcript_text,
                                is_final=True,
                                segment_id=current_segment_id,
                            )

                            entry = TranscriptEntry(
                                speaker_identity=self.participant.identity,
                                speaker_name=self.participant.name or self.participant.identity,
                                text=transcript_text,
                                timestamp=time.time(),
                                segment_id=current_segment_id,
                                is_final=True,
                            )
                            await self.insight_analyzer.add_transcript(entry)

                            if self.document_referencer:
                                await self.document_referencer.on_transcript_final(
                                    segment_id=current_segment_id,
                                    transcript=transcript_text,
                                    speaker_identity=self.participant.identity,
                                    duration_seconds=duration_seconds,
                                )

                            # Phase 4: Feed transcript to agenda tracker for topic detection
                            if self.agenda_tracker:
                                await self.agenda_tracker.process_transcript(entry)

                            # Phase 1 (Real-Time Actions): Feed transcript to action classifier for context
                            if self.action_classifier:
                                await self.action_classifier.add_transcript(entry)

                            # Phase 3 (Real-Time Actions): Feed transcript to email draft generator for context
                            if self.email_draft_generator:
                                await self.email_draft_generator.add_transcript(entry)

                            current_segment_id = None
                            self._segment_start_time = None

                    elif event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
                        transcript_text = (
                            event.alternatives[0].text if event.alternatives else ""
                        )
                        if transcript_text.strip():
                            if current_segment_id is None:
                                self._segment_counter += 1
                                current_segment_id = (
                                    f"{self.participant.identity}-{self._segment_counter}"
                                )

                            await self._publish_transcription(
                                transcript_text,
                                is_final=False,
                                segment_id=current_segment_id,
                            )

            await asyncio.gather(process_audio(), process_transcriptions())

        except asyncio.CancelledError:
            logger.info(f"Stopped transcribing {self.participant.identity}")
            raise
        except Exception as e:
            logger.error(f"Error transcribing {self.participant.identity}: {e}")

    async def _publish_transcription(self, text: str, is_final: bool, segment_id: str):
        try:
            await self.room.local_participant.send_text(
                text,
                topic=self.transcription_topic,
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

