// tutorial07.c
// A pedagogical video player that really works! Now with seeking features.
//
// This tutorial was written by Stephen Dranger (dranger@gmail.com).
//
// Code based on FFplay, Copyright (c) 2003 Fabrice Bellard, 
// and a tutorial by Martin Bohme (boehme@inb.uni-luebeckREMOVETHIS.de)
// Tested on Gentoo, CVS version 5/01/07 compiled with GCC 4.1.1
//
// Use the Makefile to build all the samples.
//
// Run using
// tutorial07 myvideofile.mpg
//
// to play the video.

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libswscale/swscale.h>
#include <libavutil/avstring.h>
#include <libavutil/time.h>

#include <SDL.h>
#include <SDL_thread.h>
#ifdef __MINGW32__
#undef main /* Prevents SDL from overriding main() */
#endif
#include <stdio.h>
#include <math.h>

#define SDL_AUDIO_BUFFER_SIZE 1024
#define MAX_AUDIO_FRAME_SIZE 192000
#define MAX_AUDIOQ_SIZE (5 * 16 * 1024)
#define MAX_VIDEOQ_SIZE (5 * 256 * 1024)
#define AV_SYNC_THRESHOLD 0.01
#define AV_NOSYNC_THRESHOLD 10.0
#define SAMPLE_CORRECTION_PERCENT_MAX 10
#define AUDIO_DIFF_AVG_NB 20
#define FF_ALLOC_EVENT   (SDL_USEREVENT)
#define FF_REFRESH_EVENT (SDL_USEREVENT + 1)
#define FF_QUIT_EVENT (SDL_USEREVENT + 2)
#define VIDEO_PICTURE_QUEUE_SIZE 1
#define DEFAULT_AV_SYNC_TYPE AV_SYNC_VIDEO_MASTER

typedef struct PacketQueue {
  AVPacketList *first_pkt, *last_pkt;
  int nb_packets;
  int size;
  SDL_mutex *mutex;
  SDL_cond *cond;
} PacketQueue;

typedef enum color{
	RGB, BW, RED, GREEN, BLUE
} Color;

typedef struct VideoPicture {
  SDL_Overlay *bmp;
  int width, height; /* source height & width */
  int allocated;
  double pts;
} VideoPicture;

typedef struct videoState {
  AVFormatContext *pFormatCtx;
  int             videoStatestream, audioStream;

  int             av_sync_type;
  double          external_clock; /* external clock base */
  int64_t         external_clock_time;
  int             seek_req;
  int             seek_flags;
  int64_t         seek_pos;

  double          audio_clock;
  AVStream        *audio_st;
  PacketQueue     audioq;
  AVFrame         audio_frame;
  uint8_t         audio_buf[(MAX_AUDIO_FRAME_SIZE * 3) / 2];
  unsigned int    audio_buf_size;
  unsigned int    audio_buf_index;
  AVPacket        audio_pkt;
  uint8_t         *audio_pkt_data;
  int             audio_pkt_size;
  int             audio_hw_buf_size;  
  double          audio_diff_cum; /* used for AV difference average computation */
  double          audio_diff_avg_coef;
  double          audio_diff_threshold;
  int             audio_diff_avg_count;
  double          frame_timer;
  double          frame_last_pts;
  double          frame_last_delay;
  double          video_clock; ///<pts of last decoded frame / predicted pts of next decoded frame
  double          video_current_pts; ///<current displayed pts (different from video_clock if frame fifos are used)
  int64_t         video_current_pts_time;  ///<time (av_gettime) at which we updated video_current_pts - used to have running video pts
  AVStream        *video_st;
  PacketQueue     videoq;
  VideoPicture    pictq[VIDEO_PICTURE_QUEUE_SIZE];
  int             pictq_size, pictq_rindex, pictq_windex;
  SDL_mutex       *pictq_mutex;
  SDL_cond        *pictq_cond;
  SDL_Thread      *parse_tid;
  SDL_Thread      *video_tid;

  char            filename[1024];
  int             quit;

  AVIOContext     *io_context;
  struct SwsContext *sws_ctx;

 /* my shanges */
  int curr;
  int savePicture;
  int isMuted;
  struct SwsContext *sws_ctxRGB;
  AVCodecContext *pCodecCtx;
  uint8_t         silence_buf[SDL_AUDIO_BUFFER_SIZE];
  Color colorState;
 /* my shanges */

} videoState;

enum {
  AV_SYNC_AUDIO_MASTER,
  AV_SYNC_VIDEO_MASTER,
  AV_SYNC_EXTERNAL_MASTER,
};

typedef enum ScreenRes {
  PLUS_SCREEN, MINUS_SCREEN
}screenRes;

SDL_Surface     *screen;

/* Since we only have one decoding thread, the Big Struct
   can be global in case we need it. */
videoState *global_video_state;
AVPacket flush_pkt;

int screenWidth;
int screenHeight;

typedef struct VideoStatesLib
{
	videoState **videoStates;
	int size;
	int playingAudio;
	int playingVideo;
} VideoStatesLib;

VideoStatesLib videoStatesLib;


void stream_seek(videoState *is, int64_t pos, int rel);

double get_video_clock(videoState *is);

 /* my shanges *///////////////////// S T A R T //////////////////////////////////

// save picture

void takePicture() {
	global_video_state->savePicture = 1;
}

// mutes

void mute() {
	global_video_state->isMuted = 1;
}

void unmute() {
	global_video_state->isMuted = 0;
}

// colors

void changeColorStateRED() {
    global_video_state->colorState=RED;
}

void changeColorStateGREEN() {
    global_video_state->colorState=GREEN;
}

void changeColorStateRGB() {
    global_video_state->colorState=RGB;
}

void changeColorStateBLUE() {
    global_video_state->colorState=BLUE;
}

void changeColorStateBW() {
	global_video_state->colorState=BW;
}

// resulution

void screenResulution(screenRes sr){
	if(sr == PLUS_SCREEN){
		screenHeight = screenHeight*1.2;
		screenWidth = screenWidth*1.2;
	}else if(sr == MINUS_SCREEN){
		screenHeight = screenHeight/1.2;
		screenWidth = screenWidth/1.2;
	}
	#ifndef __DARWIN__
	    screen = SDL_SetVideoMode(screenWidth, screenHeight, 0, 0);
	#else
		screen = SDL_SetVideoMode(screenWidth, screenHeight, 24, 0);
	#endif
}

void switchChannel(int channel) {
	double pos;
	int channelNum = channel;
	// Checks if channels in range
	if ( ((channel == 2 || channel == 5 || channel == 8) && videoStatesLib.size < 2 ) ||
		 ((channel == 3 || channel == 6 || channel == 9) && videoStatesLib.size < 3 ) )
		return;
	if ( (channel == 1 || channel == 2 || channel == 3) && channel-1 != videoStatesLib.playingAudio )
	{
    if (channel-1 == videoStatesLib.playingAudio) return;
   		printf("audio channel: %d\n",channel );
		pos = get_video_clock(videoStatesLib.videoStates[channel-1]);
		stream_seek(global_video_state, (int64_t)(pos * AV_TIME_BASE), pos);
		videoStatesLib.playingAudio = channel-1;
		return;
	}
	if ( channel > 3  && channel < 7)
	{
	   	channel = (channel-1)%3;
   	 	if (channel == videoStatesLib.playingVideo) return;
    	printf("channel video: %d\n",channelNum );
		videoStatesLib.playingVideo = channel;
		global_video_state = videoStatesLib.videoStates[videoStatesLib.playingVideo];
		return;
	}
	if ( channel > 6 && channel < 10)
	{
    	channel = (channel-1)%3;
    	if ((channel == videoStatesLib.playingVideo) && (channel == videoStatesLib.playingAudio)) return;
    	printf("pressed key: %d display video and audio num: %d\n",channelNum,channel );
		videoStatesLib.playingVideo = videoStatesLib.playingAudio = channel;
		global_video_state = videoStatesLib.videoStates[videoStatesLib.playingVideo];
		return;
	}

}
 /* my shanges *///////////////////// E N D //////////////////////////////////


void packet_queue_init(PacketQueue *q) {
  memset(q, 0, sizeof(PacketQueue));
  q->mutex = SDL_CreateMutex();
  q->cond = SDL_CreateCond();
}
int packet_queue_put(PacketQueue *q, AVPacket *pkt) {

  AVPacketList *pkt1;
  if(pkt != &flush_pkt && av_dup_packet(pkt) < 0) {
    return -1;
  }
  pkt1 = av_malloc(sizeof(AVPacketList));
  if (!pkt1)
    return -1;
  pkt1->pkt = *pkt;
  pkt1->next = NULL;
  
  SDL_LockMutex(q->mutex);

  if (!q->last_pkt)
    q->first_pkt = pkt1;
  else
    q->last_pkt->next = pkt1;
  q->last_pkt = pkt1;
  q->nb_packets++;
  q->size += pkt1->pkt.size;
  SDL_CondSignal(q->cond);
  
  SDL_UnlockMutex(q->mutex);
  return 0;
}
static int packet_queue_get(PacketQueue *q, AVPacket *pkt, int block)
{
  AVPacketList *pkt1;
  int ret;

  SDL_LockMutex(q->mutex);
  
  for(;;) {
    
    if(global_video_state->quit) {
      ret = -1;
      break;
    }

    pkt1 = q->first_pkt;
    if (pkt1) {
      q->first_pkt = pkt1->next;
      if (!q->first_pkt)
	q->last_pkt = NULL;
      q->nb_packets--;
      q->size -= pkt1->pkt.size;
      *pkt = pkt1->pkt;
      av_free(pkt1);
      ret = 1;
      break;
    } else if (!block) {
      ret = 0;
      break;
    } else {
      SDL_CondWait(q->cond, q->mutex);
    }
  }
  SDL_UnlockMutex(q->mutex);
  return ret;
}
static void packet_queue_flush(PacketQueue *q) {
  AVPacketList *pkt, *pkt1;

  SDL_LockMutex(q->mutex);
  for(pkt = q->first_pkt; pkt != NULL; pkt = pkt1) {
    pkt1 = pkt->next;
    av_free_packet(&pkt->pkt);
    av_freep(&pkt);
  }
  q->last_pkt = NULL;
  q->first_pkt = NULL;
  q->nb_packets = 0;
  q->size = 0;
  SDL_UnlockMutex(q->mutex);
}
double get_audio_clock(videoState *is) {
  double pts;
  int hw_buf_size, bytes_per_sec, n;

  pts = is->audio_clock; /* maintained in the audio thread */
  hw_buf_size = is->audio_buf_size - is->audio_buf_index;
  bytes_per_sec = 0;
  n = is->audio_st->codec->channels * 2;
  if(is->audio_st) {
    bytes_per_sec = is->audio_st->codec->sample_rate * n;
  }
  if(bytes_per_sec) {
    pts -= (double)hw_buf_size / bytes_per_sec;
  }
  return pts;
}
double get_video_clock(videoState *is) {
  double delta;

  delta = (av_gettime() - is->video_current_pts_time) / 1000000.0;
  return is->video_current_pts + delta;
}
double get_external_clock(videoState *is) {
  return av_gettime() / 1000000.0;
}
double get_master_clock(videoState *is) {
  if(is->av_sync_type == AV_SYNC_VIDEO_MASTER) {
    return get_video_clock(is);
  } else if(is->av_sync_type == AV_SYNC_AUDIO_MASTER) {
    return get_audio_clock(is);
  } else {
    return get_external_clock(is);
  }
}
/* Add or subtract samples to get a better sync, return new
   audio buffer size */
int synchronize_audio(videoState *is, short *samples,
		      int samples_size, double pts) {
  int n;
  double ref_clock;
  
  n = 2 * is->audio_st->codec->channels;
  
  if(is->av_sync_type != AV_SYNC_AUDIO_MASTER) {
    double diff, avg_diff;
    int wanted_size, min_size, max_size /*, nb_samples */;
    
    ref_clock = get_master_clock(is);
    diff = get_audio_clock(is) - ref_clock;

    if(diff < AV_NOSYNC_THRESHOLD) {
      // accumulate the diffs
      is->audio_diff_cum = diff + is->audio_diff_avg_coef
	* is->audio_diff_cum;
      if(is->audio_diff_avg_count < AUDIO_DIFF_AVG_NB) {
	is->audio_diff_avg_count++;
      } else {
	avg_diff = is->audio_diff_cum * (1.0 - is->audio_diff_avg_coef);
	if(fabs(avg_diff) >= is->audio_diff_threshold) {
	  wanted_size = samples_size + ((int)(diff * is->audio_st->codec->sample_rate) * n);
	  min_size = samples_size * ((100 - SAMPLE_CORRECTION_PERCENT_MAX) / 100);
	  max_size = samples_size * ((100 + SAMPLE_CORRECTION_PERCENT_MAX) / 100);
	  if(wanted_size < min_size) {
	    wanted_size = min_size;
	  } else if (wanted_size > max_size) {
	    wanted_size = max_size;
	  }
	  if(wanted_size < samples_size) {
	    /* remove samples */
	    samples_size = wanted_size;
	  } else if(wanted_size > samples_size) {
	    uint8_t *samples_end, *q;
	    int nb;

	    /* add samples by copying final sample*/
	    nb = (samples_size - wanted_size);
	    samples_end = (uint8_t *)samples + samples_size - n;
	    q = samples_end + n;
	    while(nb > 0) {
	      memcpy(q, samples_end, n);
	      q += n;
	      nb -= n;
	    }
	    samples_size = wanted_size;
	  }
	}
      }
    } else {
      /* difference is TOO big; reset diff stuff */
      is->audio_diff_avg_count = 0;
      is->audio_diff_cum = 0;
    }
  }
  return samples_size;
}

int audio_decode_frame(videoState *is, double *pts_ptr) {

  int len1, data_size = 0, n;
  AVPacket *pkt = &is->audio_pkt;
  double pts;

  for(;;) {
    while(is->audio_pkt_size > 0) {
      int got_frame = 0;
      len1 = avcodec_decode_audio4(is->audio_st->codec, &is->audio_frame, &got_frame, pkt);
      if(len1 < 0) {
	/* if error, skip frame */
	is->audio_pkt_size = 0;
	break;
      }
      if (got_frame)
      {
          data_size = 
            av_samples_get_buffer_size
            (
                NULL, 
                is->audio_st->codec->channels,
                is->audio_frame.nb_samples,
                is->audio_st->codec->sample_fmt,
                1
            );
          memcpy(is->audio_buf, is->audio_frame.data[0], data_size);
      }
      is->audio_pkt_data += len1;
      is->audio_pkt_size -= len1;
      if(data_size <= 0) {
	/* No data yet, get more frames */
	continue;
      }
      pts = is->audio_clock;
      *pts_ptr = pts;
      n = 2 * is->audio_st->codec->channels;
      is->audio_clock += (double)data_size /
	(double)(n * is->audio_st->codec->sample_rate);

      /* We have data, return it and come back for more later */
      return data_size;
    }
    if(pkt->data)
      av_free_packet(pkt);

    if(is->quit) {
      return -1;
    }
    /* next packet */
    if(packet_queue_get(&is->audioq, pkt, 1) < 0) {
      return -1;
    }
    if(pkt->data == flush_pkt.data) {
      avcodec_flush_buffers(is->audio_st->codec);
      continue;
    }
    is->audio_pkt_data = pkt->data;
    is->audio_pkt_size = pkt->size;
    /* if update, update the audio clock w/pts */
    if(pkt->pts != AV_NOPTS_VALUE) {
      is->audio_clock = av_q2d(is->audio_st->time_base)*pkt->pts;
    }
  }
}

 /* my shanges *///////////////////// S T A R T //////////////////////////////////
void audio_callback(void *userdata, Uint8 *stream, int len) {

  videoState *is = (videoState *)userdata;
  int len1, audio_size;
  double pts;
  if (is->curr != videoStatesLib.playingAudio)
  {
  	is = videoStatesLib.videoStates[videoStatesLib.playingAudio];
  }
  while(len > 0) {
    if(is->audio_buf_index >= is->audio_buf_size) {
      /* We have already sent all our data; get more */
      audio_size = audio_decode_frame(is, &pts);
      if(audio_size < 0) {
	/* If error, output silence */
	is->audio_buf_size = 1024;
	memset(is->audio_buf, 0, is->audio_buf_size);
      } else {
	audio_size = synchronize_audio(is, (int16_t *)is->audio_buf,
				       audio_size, pts);
	is->audio_buf_size = audio_size;
      }
      is->audio_buf_index = 0;
    }
    len1 = is->audio_buf_size - is->audio_buf_index;
    if(len1 > len)
      len1 = len;

    // chacks for muting
    if (!is->isMuted) 
      memcpy(stream, (uint8_t *)is->audio_buf + is->audio_buf_index, len1);
     else{
      memset(stream, is->silence_buf[0], len1);
      if (!is->isMuted)
          SDL_MixAudio(stream, (uint8_t *)is->audio_buf + is->audio_buf_index, len1, 0);
    }

    len -= len1;
    stream += len1;
    is->audio_buf_index += len1;
  }
}
/* my shanges *///////////////////// E N D //////////////////////////////////

static Uint32 sdl_refresh_timer_cb(Uint32 interval, void *opaque) {
  SDL_Event event;
  event.type = FF_REFRESH_EVENT;
  event.user.data1 = opaque;
  SDL_PushEvent(&event);
  return 0; /* 0 means stop timer */
}

/* schedule a video refresh in 'delay' ms */
static void schedule_refresh(videoState *is, int delay) {
  SDL_AddTimer(delay, sdl_refresh_timer_cb, is);
}

void video_display(videoState *is) {

  SDL_Rect rect;
  VideoPicture *vp;
  //AVPicture pict;
  float aspect_ratio;
  int w, h, x, y;
  //int i;
  if (is->curr == videoStatesLib.playingVideo) 
  {
  	vp = &is->pictq[is->pictq_rindex];
  	if(vp->bmp) {
    if(is->video_st->codec->sample_aspect_ratio.num == 0) {
      aspect_ratio = 0;
    } else {
      aspect_ratio = av_q2d(is->video_st->codec->sample_aspect_ratio) *
	is->video_st->codec->width / is->video_st->codec->height;
    }
    if(aspect_ratio <= 0.0) {
      aspect_ratio = (float)is->video_st->codec->width /
	(float)is->video_st->codec->height;
    }
    h = screen->h;
    w = ((int)rint(h * aspect_ratio)) & -3;
    if(w > screen->w) {
      w = screen->w;
      h = ((int)rint(w / aspect_ratio)) & -3;
    }
    x = (screen->w - w) / 2;
    y = (screen->h - h) / 2;
    
    rect.x = x;
    rect.y = y;
    rect.w = w;
    rect.h = h;
    SDL_DisplayYUVOverlay(vp->bmp, &rect);
  }
  }

}

void video_refresh_timer(void *userdata) {

  videoState *is = (videoState *)userdata;
  VideoPicture *vp;
  double actual_delay, delay, sync_threshold, ref_clock, diff;
  
  if(is->video_st) {
    if(is->pictq_size == 0) {
      schedule_refresh(is, 1);
    } else {
      vp = &is->pictq[is->pictq_rindex];

      is->video_current_pts = vp->pts;
      is->video_current_pts_time = av_gettime();

      delay = vp->pts - is->frame_last_pts; /* the pts from last time */
      if(delay <= 0 || delay >= 1.0) {
	/* if incorrect delay, use previous one */
	delay = is->frame_last_delay;
      }
      /* save for next time */
      is->frame_last_delay = delay;
      is->frame_last_pts = vp->pts;

      /* update delay to sync to audio if not master source */
      if(is->av_sync_type != AV_SYNC_VIDEO_MASTER) {
	ref_clock = get_master_clock(is);
	diff = vp->pts - ref_clock;
	
	/* Skip or repeat the frame. Take delay into account
	   FFPlay still doesn't "know if this is the best guess." */
	sync_threshold = (delay > AV_SYNC_THRESHOLD) ? delay : AV_SYNC_THRESHOLD;
	if(fabs(diff) < AV_NOSYNC_THRESHOLD) {
	  if(diff <= -sync_threshold) {
	    delay = 0;
	  } else if(diff >= sync_threshold) {
	    delay = 2 * delay;
	  }
	}
      }

      is->frame_timer += delay;
      /* computer the REAL delay */
      actual_delay = is->frame_timer - (av_gettime() / 1000000.0);
      if(actual_delay < 0.010) {
	/* Really it should skip the picture instead */
	actual_delay = 0.010;
      }
      schedule_refresh(is, (int)(actual_delay * 1000 + 0.5));

      /* show the picture! */
      video_display(is);
      
      /* update queue for next picture! */
      if(++is->pictq_rindex == VIDEO_PICTURE_QUEUE_SIZE) {
	is->pictq_rindex = 0;
      }
      SDL_LockMutex(is->pictq_mutex);
      is->pictq_size--;
      SDL_CondSignal(is->pictq_cond);
      SDL_UnlockMutex(is->pictq_mutex);
    }
  } else {
    schedule_refresh(is, 100);
  }
}
      
void alloc_picture(void *userdata) {

  videoState *is = (videoState *)userdata;
  VideoPicture *vp;

  vp = &is->pictq[is->pictq_windex];
  if(vp->bmp) {
    // we already have one make another, bigger/smaller
    SDL_FreeYUVOverlay(vp->bmp);
  }
  // Allocate a place to put our YUV image on that screen
  vp->bmp = SDL_CreateYUVOverlay(is->video_st->codec->width,
				 is->video_st->codec->height,
				 SDL_YV12_OVERLAY,
				 screen);
  vp->width = is->video_st->codec->width;
  vp->height = is->video_st->codec->height;
  
  SDL_LockMutex(is->pictq_mutex);
  vp->allocated = 1;
  SDL_CondSignal(is->pictq_cond);
  SDL_UnlockMutex(is->pictq_mutex);

}

/* my shanges *///////////////////// S T A R T //////////////////////////////////
int queue_picture(videoState *is, AVFrame *pFrame, double pts) {

  VideoPicture *vp;
  AVPicture pict;

  /* wait until we have space for a new pic */
  SDL_LockMutex(is->pictq_mutex);
  while(is->pictq_size >= VIDEO_PICTURE_QUEUE_SIZE &&
	!is->quit) {
    SDL_CondWait(is->pictq_cond, is->pictq_mutex);
  }
  SDL_UnlockMutex(is->pictq_mutex);

  if(is->quit)
    return -1;

  // windex is set to 0 initially
  vp = &is->pictq[is->pictq_windex];

  /* allocate or resize the buffer! */
  if(!vp->bmp ||
     vp->width != is->video_st->codec->width ||
     vp->height != is->video_st->codec->height) {
    SDL_Event event;

    vp->allocated = 0;
    /* we have to do it in the main thread */
    event.type = FF_ALLOC_EVENT;
    event.user.data1 = is;
    SDL_PushEvent(&event);

    /* wait until we have a picture allocated */
    SDL_LockMutex(is->pictq_mutex);
    while(!vp->allocated && !is->quit) {
      SDL_CondWait(is->pictq_cond, is->pictq_mutex);
    }
    SDL_UnlockMutex(is->pictq_mutex);
    if(is->quit) {
      return -1;
    }
  }
  /* We have a place to put our picture on the queue */
  /* If we are skipping a frame, do we set this to null 
     but still return vp->allocated = 1? */


  if(vp->bmp) {

    SDL_LockYUVOverlay(vp->bmp);
    
    //dst_pix_fmt = PIX_FMT_YUV420P;
    /* point pict at the queue */

    pict.data[0] = vp->bmp->pixels[0];
    pict.data[1] = vp->bmp->pixels[2];
    pict.data[2] = vp->bmp->pixels[1];
    
    pict.linesize[0] = vp->bmp->pitches[0];
    pict.linesize[1] = vp->bmp->pitches[2];
    pict.linesize[2] = vp->bmp->pitches[1];
    
    if(is->colorState != BW && is->colorState != RGB){
    struct SwsContext * ctx = sws_getContext(is->video_st->codec->width, is->video_st->codec->height,
        PIX_FMT_RGB24, is->video_st->codec->width, is->video_st->codec->height,
        PIX_FMT_YUV420P, 0, 0, 0, 0); 
    AVFrame *pFrameRGB = NULL;
    // Allocate video frame
    pFrameRGB=av_frame_alloc();
    if(pFrameRGB==NULL)
        return -1;
    uint8_t *buffer = NULL;
    int numBytes;
    // Determine required buffer size and allocate buffer
    numBytes=avpicture_get_size(PIX_FMT_RGB24, is->pCodecCtx->width,
        is->pCodecCtx->height);
    buffer=(uint8_t *)av_malloc(numBytes*sizeof(uint8_t));
    avpicture_fill((AVPicture *)pFrameRGB, buffer, PIX_FMT_RGB24,
        is->pCodecCtx->width, is->pCodecCtx->height);
    sws_scale(is->sws_ctxRGB, (uint8_t const * const *)pFrame->data,
        pFrame->linesize, 0, is->pCodecCtx->height,
        pFrameRGB->data, pFrameRGB->linesize);
     int y,x;
     int tmp1,tmp2;
    for(y=0;y<is->video_st->codec->height;y++){
      for(x=0;x<is->video_st->codec->width*3;x+=3){
      	  if(is->colorState==BLUE){
		  		tmp1=0;tmp2=1;
		  }else if(is->colorState==RED){
		  		tmp1=1,tmp2=2;
		  }else if(is->colorState==GREEN){
		  		tmp1=0;tmp2=2;
		  }
		  pFrameRGB->data[0][y*pFrameRGB->linesize[0]+x+tmp1] =0;
		  pFrameRGB->data[0][y*pFrameRGB->linesize[0]+x+tmp2] =0;
      }
    }
    sws_scale(ctx, (uint8_t const * const *)pFrameRGB->data,
    pFrameRGB->linesize,0,is->video_st->codec->height,
    pFrame->data,pFrame->linesize);
}  

    // Convert the image into YUV format that SDL uses
    sws_scale
    (
        is->sws_ctx,
        (uint8_t const * const *)pFrame->data,
        pFrame->linesize,
        0, 
        is->video_st->codec->height, 
        pict.data, 
        pict.linesize
    );
    
    SDL_UnlockYUVOverlay(vp->bmp);
    vp->pts = pts;

    /* now we inform our display thread that we have a pic ready */
    if(++is->pictq_windex == VIDEO_PICTURE_QUEUE_SIZE) {
      is->pictq_windex = 0;
    }
    SDL_LockMutex(is->pictq_mutex);
    is->pictq_size++;
    SDL_UnlockMutex(is->pictq_mutex);
  }
  return 0;
}
/* my shanges *///////////////////// E N D //////////////////////////////////



double synchronize_video(videoState *is, AVFrame *src_frame, double pts) {

  double frame_delay;

  if(pts != 0) {
    /* if we have pts, set video clock to it */
    is->video_clock = pts;
  } else {
    /* if we aren't given a pts, set it to the clock */
    pts = is->video_clock;
  }
  /* update the video clock */
  frame_delay = av_q2d(is->video_st->codec->time_base);
  /* if we are repeating a frame, adjust clock accordingly */
  frame_delay += src_frame->repeat_pict * (frame_delay * 0.5);
  is->video_clock += frame_delay;
  return pts;
}

void saveFrame(AVFrame *pFrame, int width, int height){
    FILE *pFile;
    char szFilename[32];
    int  y;

    pFile=fopen("$tempname$.ppm", "wb");
    if(pFile==NULL)
        return;

    // Write header
    fprintf(pFile, "P6\n%d %d\n255\n", width, height);

    // Write pixel data
    for(y=0; y<height; y++)
        fwrite(pFrame->data[0]+y*pFrame->linesize[0], 1, width*3, pFile);

    // Close file
    fclose(pFile);
}

uint64_t global_video_pkt_pts = AV_NOPTS_VALUE;

/* These are called whenever we allocate a frame
 * buffer. We use this to store the global_pts in
 * a frame at the time it is allocated.
 */
int our_get_buffer(struct AVCodecContext *c, AVFrame *pic) {
  int ret = avcodec_default_get_buffer(c, pic);
  uint64_t *pts = av_malloc(sizeof(uint64_t));
  *pts = global_video_pkt_pts;
  pic->opaque = pts;
  return ret;
}
void our_release_buffer(struct AVCodecContext *c, AVFrame *pic) {
  if(pic) av_freep(&pic->opaque);
  avcodec_default_release_buffer(c, pic);
}

int video_thread(void *arg) {
  videoState *is = (videoState *)arg;
  AVPacket pkt1, *packet = &pkt1;
  int frameFinished;
  AVFrame *pFrame = NULL;
  double pts;

  pFrame = av_frame_alloc();

  for(;;) {
    if(packet_queue_get(&is->videoq, packet, 1) < 0) {
      // means we quit getting packets
      break;
    }
    if(packet->data == flush_pkt.data) {
      avcodec_flush_buffers(is->video_st->codec);
      continue;
    }
    pts = 0;

    // Save global pts to be stored in pFrame in first call
    global_video_pkt_pts = packet->pts;

    // Decode video frame
    avcodec_decode_video2(is->video_st->codec, pFrame, &frameFinished, 
				packet);
    if(packet->dts == AV_NOPTS_VALUE 
       && pFrame->opaque && *(uint64_t*)pFrame->opaque != AV_NOPTS_VALUE) {
      pts = *(uint64_t *)pFrame->opaque;
    } else if(packet->dts != AV_NOPTS_VALUE) {
      pts = packet->dts;
    } else {
      pts = 0;
    }
    pts *= av_q2d(is->video_st->time_base);

    // Did we get a video frame?
    if(frameFinished) {

		if (is->savePicture){
		    AVFrame *pFrameRGB=av_frame_alloc();
		    if(pFrameRGB==NULL)
		        return -1;
		    int numBytes=avpicture_get_size(PIX_FMT_RGB24, is->pCodecCtx->width,
		                                    is->pCodecCtx->height);

		    uint8_t *buffer=(uint8_t *)av_malloc(numBytes*sizeof(uint8_t));
		    avpicture_fill((AVPicture *)pFrameRGB, buffer, PIX_FMT_RGB24,
		                   is->pCodecCtx->width, is->pCodecCtx->height);
		    sws_scale(is->sws_ctxRGB, (uint8_t const * const *)pFrame->data,
		              pFrame->linesize, 0, is->video_st->codec->height,
		              pFrameRGB->data, pFrameRGB->linesize);
		    saveFrame(pFrameRGB, is->video_st->codec->width, is->video_st->codec->height);
		    is->savePicture=0;
		}

	    // coloring white&blace
      	if (is->colorState == BW){        
        	memset(pFrame->data[1],128,(pFrame->linesize[1]*pFrame->height)/2);
       	 	memset(pFrame->data[2],128,(pFrame->linesize[2]*pFrame->height)/2);
        }

		// sync video
        if (videoStatesLib.playingVideo == videoStatesLib.playingAudio)
        {
        	pts = synchronize_video(is, pFrame, pts);
        }

	    if(queue_picture(is, pFrame, pts) < 0) {
			break;
	    }
    }
    av_free_packet(packet);
  }
  av_free(pFrame);
  return 0;
}

int stream_component_open(videoState *is, int stream_index) {

  AVFormatContext *pFormatCtx = is->pFormatCtx;
  AVCodecContext *codecCtx = NULL;
  AVCodec *codec = NULL;
  AVDictionary *optionsDict = NULL;
  SDL_AudioSpec wanted_spec, spec;

  if(stream_index < 0 || stream_index >= pFormatCtx->nb_streams) {
    return -1;
  }

  // Get a pointer to the codec context for the video stream
  codecCtx = pFormatCtx->streams[stream_index]->codec;

  if (codecCtx->sample_fmt == AV_SAMPLE_FMT_S16P)
    codecCtx->request_sample_fmt = AV_SAMPLE_FMT_S16;

  if(codecCtx->codec_type == AVMEDIA_TYPE_AUDIO) {
    // Set audio settings from codec info
    wanted_spec.freq = codecCtx->sample_rate;
    wanted_spec.format = AUDIO_S16SYS;
    wanted_spec.channels = codecCtx->channels;
    wanted_spec.silence = 0;
    wanted_spec.samples = SDL_AUDIO_BUFFER_SIZE;
    wanted_spec.callback = audio_callback;
    wanted_spec.userdata = is;
    if (is->curr == 1)
      if(SDL_OpenAudio(&wanted_spec, &spec) < 0) {
        fprintf(stderr, "SDL_OpenAudio: %s\n", SDL_GetError());
        return -1;
    }
    is->audio_hw_buf_size = spec.size;
  }
  codec = avcodec_find_decoder(codecCtx->codec_id);
  if(!codec || (avcodec_open2(codecCtx, codec, &optionsDict) < 0)) {
    fprintf(stderr, "Unsupported codec!\n");
    return -1;
  }

  switch(codecCtx->codec_type) {
  case AVMEDIA_TYPE_AUDIO:
    is->audioStream = stream_index;
    is->audio_st = pFormatCtx->streams[stream_index];
    is->audio_buf_size = 0;
    is->audio_buf_index = 0;
    
    /* averaging filter for audio sync */
    is->audio_diff_avg_coef = exp(log(0.01 / AUDIO_DIFF_AVG_NB));
    is->audio_diff_avg_count = 0;
    /* Correct audio only if larger error than this */
    is->audio_diff_threshold = 2.0 * SDL_AUDIO_BUFFER_SIZE / codecCtx->sample_rate;

    memset(&is->audio_pkt, 0, sizeof(is->audio_pkt));
    packet_queue_init(&is->audioq);
    SDL_PauseAudio(0);
    break;
  case AVMEDIA_TYPE_VIDEO:
    is->videoStatestream = stream_index;
    is->video_st = pFormatCtx->streams[stream_index];

    is->frame_timer = (double)av_gettime() / 1000000.0;
    is->frame_last_delay = 40e-3;
    is->video_current_pts_time = av_gettime();

    packet_queue_init(&is->videoq);
    is->video_tid = SDL_CreateThread(video_thread, is);
    is->sws_ctx =
        sws_getContext
        (
            is->video_st->codec->width,
            is->video_st->codec->height,
            is->video_st->codec->pix_fmt,
            is->video_st->codec->width,
            is->video_st->codec->height,
            PIX_FMT_YUV420P, 
            SWS_BILINEAR, 
            NULL, 
            NULL, 
            NULL
        );
        is->pCodecCtx = avcodec_alloc_context3(codec);
        if(avcodec_copy_context(is->pCodecCtx, is->video_st->codec) != 0){
            fprintf(stderr, "Couldn't copy codec context");
            return -1; // Error copying codec context
        }
         is->sws_ctxRGB = sws_getContext(is->pCodecCtx->width,
                                        is->pCodecCtx->height,
                                        is->pCodecCtx->pix_fmt,
                                        is->pCodecCtx->width,
                                        is->pCodecCtx->height,
                                        PIX_FMT_RGB24,
                                        SWS_BILINEAR,
                                        NULL,
                                        NULL,
                                        NULL
                                       );

    break;
  default:
    break;
  }

  return 0;
}

int decode_interrupt_cb(void *opaque) {
  return (global_video_state && global_video_state->quit);
}
int decode_thread(void *arg) {

  videoState *is = (videoState *)arg;
  AVFormatContext *pFormatCtx = NULL;
  AVPacket pkt1, *packet = &pkt1;

  AVDictionary *io_dict = NULL;
  AVIOInterruptCB callback;

  int video_index = -1;
  int audio_index = -1;
  int i;

  is->videoStatestream=-1;
  is->audioStream=-1;

  global_video_state = is;
  // will interrupt blocking functions if we quit!
  callback.callback = decode_interrupt_cb;
  callback.opaque = is;
  if (avio_open2(&is->io_context, is->filename, 0, &callback, &io_dict))
  {
    fprintf(stderr, "Unable to open I/O for %s\n", is->filename);
    return -1;
  }

  // Open video file
  if(avformat_open_input(&pFormatCtx, is->filename, NULL, NULL)!=0)
    return -1; // Couldn't open file

  is->pFormatCtx = pFormatCtx;
  
  // Retrieve stream information
  if(avformat_find_stream_info(pFormatCtx, NULL)<0)
    return -1; // Couldn't find stream information
  
  // Dump information about file onto standard error
  av_dump_format(pFormatCtx, 0, is->filename, 0);
  
  // Find the first video stream
  for(i=0; i<pFormatCtx->nb_streams; i++) {
    if(pFormatCtx->streams[i]->codec->codec_type==AVMEDIA_TYPE_VIDEO &&
       video_index < 0) {
      video_index=i;
    }
    if(pFormatCtx->streams[i]->codec->codec_type==AVMEDIA_TYPE_AUDIO &&
       audio_index < 0) {
      audio_index=i;
    }
  }
  if(audio_index >= 0) {
    stream_component_open(is, audio_index);
  }
  if(video_index >= 0) {
    stream_component_open(is, video_index);
  }   

  if(is->videoStatestream < 0 || is->audioStream < 0) {
    fprintf(stderr, "%s: could not open codecs\n", is->filename);
    goto fail;
  }

  // main decode loop

  for(;;) {
    if(is->quit) {
      break;
    }
    // seek stuff goes here
    if(is->seek_req) {
      int stream_index= -1;
      int64_t seek_target = is->seek_pos;

      if     (is->videoStatestream >= 0) stream_index = is->videoStatestream;
      else if(is->audioStream >= 0) stream_index = is->audioStream;

      if(stream_index>=0){
	seek_target= av_rescale_q(seek_target, AV_TIME_BASE_Q, pFormatCtx->streams[stream_index]->time_base);
      }
      if(av_seek_frame(is->pFormatCtx, stream_index, seek_target, is->seek_flags) < 0) {
	fprintf(stderr, "%s: error while seeking\n", is->pFormatCtx->filename);
      } else {
	if(is->audioStream >= 0) {
	  packet_queue_flush(&is->audioq);
	  packet_queue_put(&is->audioq, &flush_pkt);
	}
	if(is->videoStatestream >= 0) {
	  packet_queue_flush(&is->videoq);
	  packet_queue_put(&is->videoq, &flush_pkt);
	}
      }
      is->seek_req = 0;
    }

    if(is->audioq.size > MAX_AUDIOQ_SIZE ||
       is->videoq.size > MAX_VIDEOQ_SIZE) {
      SDL_Delay(10);
      //continue;
    }
    if(av_read_frame(is->pFormatCtx, packet) < 0) {
      if(is->pFormatCtx->pb->error == 0) {
	SDL_Delay(100); /* no error; wait for user input */
	continue;
      } else {
	break;
      }
    }
    // Is this a packet from the video stream?
    if(packet->stream_index == is->videoStatestream) {
      packet_queue_put(&is->videoq, packet);
    } else if(packet->stream_index == is->audioStream) {
      packet_queue_put(&is->audioq, packet);
    } else {
      av_free_packet(packet);
    }
  }
  /* all done - wait for it */
  while(!is->quit) {
    SDL_Delay(100);
  }
 fail:
  {
    SDL_Event event;
    event.type = FF_QUIT_EVENT;
    event.user.data1 = is;
    SDL_PushEvent(&event);
  }
  return 0;
}

void stream_seek(videoState *is, int64_t pos, int rel) {

  if(!is->seek_req) {
    is->seek_pos = pos;
    is->seek_flags = rel < 0 ? AVSEEK_FLAG_BACKWARD : 0;
    is->seek_req = 1;
  }
}


int main(int argc, char *argv[]) {

  SDL_Event       event;
  screenHeight = 480;
  screenWidth = 640;

  if(argc < 2) {
    fprintf(stderr, "Usage: test <file>\n");
    exit(1);
  }
  if (argc > 4)
  {
  	fprintf(stderr, "error on over loading videos, limit is 3\n");
  	exit(1);
  }

  int i=0;
  videoStatesLib.size = argc-1;
  videoStatesLib.videoStates = malloc(sizeof(videoState*)*videoStatesLib.size);

  if (!videoStatesLib.videoStates) {
  	fprintf(stderr, "Error allocating\n");
  	exit(1);
  }
  for (i = 0; i < videoStatesLib.size; ++i)
  {
  	videoStatesLib.videoStates[i] = av_mallocz(sizeof(videoState));
  	if (!videoStatesLib.videoStates[i])
  	{
  		fprintf(stderr, "Error allocating video\n");
  		exit(1);
  	}
  	videoStatesLib.videoStates[i]->curr = i;
  }

  av_register_all();
  if(SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO | SDL_INIT_TIMER)) {
    fprintf(stderr, "Error initialize SDL - %s\n", SDL_GetError());
    exit(1);
  }

  #ifndef __DARWIN__
	screen = SDL_SetVideoMode(640, 480, 0, 0);
  #else
	screen = SDL_SetVideoMode(640, 480, 24, 0);
  #endif
  if(!screen) {
	fprintf(stderr, "SDL: could not set video mode - exiting\n");
    exit(1);
  }

  for (i = 0; i < videoStatesLib.size; ++i)
  {
	  av_strlcpy(videoStatesLib.videoStates[i]->filename, argv[i+1], 1024);
	  videoStatesLib.videoStates[i]->pictq_mutex = SDL_CreateMutex();
	  videoStatesLib.videoStates[i]->pictq_cond = SDL_CreateCond();
	  schedule_refresh(videoStatesLib.videoStates[i], 40);
	  videoStatesLib.videoStates[i]->av_sync_type = DEFAULT_AV_SYNC_TYPE;
	  videoStatesLib.videoStates[i]->parse_tid = SDL_CreateThread(decode_thread, videoStatesLib.videoStates[i]);
	  if(!videoStatesLib.videoStates[i]->parse_tid) {
	    av_free(videoStatesLib.videoStates[i]);
	    return -1;
	  }
  }
  global_video_state = videoStatesLib.videoStates[0];
  videoStatesLib.playingVideo = videoStatesLib.playingAudio = 0;

  av_init_packet(&flush_pkt);
  flush_pkt.data = (unsigned char *)"FLUSH";
  
  for(;;) {
    double incr, pos;
    SDL_WaitEvent(&event);
    switch(event.type) {
    case SDL_KEYDOWN:
      switch(event.key.keysym.sym) {
      case SDLK_LEFT:
	incr = -10.0;
	goto do_seek;
      case SDLK_RIGHT:
	incr = 10.0;
	goto do_seek;
      case SDLK_UP:
	incr = 60.0;
	goto do_seek;
      case SDLK_DOWN:
	incr = -60.0;
	goto do_seek;
      do_seek:
	if(global_video_state) {
	  pos = get_master_clock(global_video_state);
	  pos += incr;
	  stream_seek(global_video_state, (int64_t)(pos * AV_TIME_BASE), incr);
	}
	break;
    case SDLK_KP_PLUS:
      screenResulution(PLUS_SCREEN);
      break;
    case SDLK_KP_MINUS:
      screenResulution(MINUS_SCREEN);
    case 'c':
      changeColorStateRGB();
      break;
    case 'g':
      changeColorStateGREEN();
      break;
    case 'b':
      changeColorStateBLUE();
      break;
    case 'r':
      changeColorStateRED();
      break;
    case 'w':
      changeColorStateBW();
      break;                        
    case 'm':
      mute();
      break;
    case 'v':
      unmute();
      break;
    case 'p':
      takePicture();
      break;
    case '1':
      switchChannel(event.key.keysym.sym-'0');
      break;
    case '2':
      switchChannel(event.key.keysym.sym-'0');
      break;
    case '3':
      switchChannel(event.key.keysym.sym-'0');
      break;
    case '4':
      switchChannel(event.key.keysym.sym-'0');
      break;
    case '5':
      switchChannel(event.key.keysym.sym-'0');
      break;
    case '6':
      switchChannel(event.key.keysym.sym-'0');
      break;
    case '7':
      switchChannel(event.key.keysym.sym-'0');
      break;
    case '8':
      switchChannel(event.key.keysym.sym-'0');
      break;                                    
    case '9':
      switchChannel(event.key.keysym.sym-'0');
      break;
      default:
	break;
      }
      break;
    case FF_QUIT_EVENT:
    case SDL_QUIT:
      videoStatesLib.videoStates[0]->quit = 1;
      /*
       * If the video has finished playing, then both the picture and
       * audio queues are waiting for more data.  Make them stop
       * waiting and terminate normally.
       */
      SDL_CondSignal(videoStatesLib.videoStates[0]->audioq.cond);
      SDL_CondSignal(videoStatesLib.videoStates[0]->videoq.cond);
      SDL_Quit();
      exit(0);
      break;
    case FF_ALLOC_EVENT:
      alloc_picture(event.user.data1);
      break;
    case FF_REFRESH_EVENT:
      video_refresh_timer(event.user.data1);
      break;
    default:
      break;
    }
  }
  return 0;
}
