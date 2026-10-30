#include <libheif/heif.h>
#include <libheif/heif_tiling.h>
#include <algorithm>
#include <string>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
extern "C" int wave_heif_decode_tiled_rgb(const char* path,int workers,unsigned char** output,int* out_w,int* out_h,char* message,size_t message_size){
 if(!path||!output||!out_w||!out_h||workers<1)return 1;*output=nullptr;
 heif_context*c=heif_context_alloc();if(!c)return 2;heif_error e=heif_context_read_from_file(c,path,nullptr);if(e.code){if(message&&message_size)snprintf(message,message_size,"read: %s",e.message?e.message:"error");heif_context_free(c);return 3;}
 heif_image_handle*h=nullptr;e=heif_context_get_primary_image_handle(c,&h);if(e.code){if(message&&message_size)snprintf(message,message_size,"handle: %s",e.message?e.message:"error");heif_context_free(c);return 4;}
 int w=heif_image_handle_get_width(h),hh=heif_image_handle_get_height(h);heif_image_tiling t{};t.version=1;e=heif_image_handle_get_image_tiling(h,1,&t);if(e.code||!t.num_columns||!t.num_rows||!t.tile_width||!t.tile_height){if(message&&message_size)snprintf(message,message_size,"tiling metadata unavailable");heif_image_handle_release(h);heif_context_free(c);return 5;}
 int n=t.num_columns*t.num_rows,sx=t.left_offset?((int)t.left_offset-(int)t.tile_width):0,sy=t.top_offset?((int)t.top_offset-(int)t.tile_height):0;size_t bytes=(size_t)w*hh*3;unsigned char*dst=(unsigned char*)malloc(bytes);if(!dst){heif_image_handle_release(h);heif_context_free(c);return 7;}memset(dst,0,bytes);std::atomic<int>next{0},failed{0};int count=std::min(workers,n);std::vector<std::thread>threads;std::string filename(path);
 for(int q=0;q<count;q++)threads.emplace_back([&,q]{heif_context*wc=heif_context_alloc();if(!wc){failed.store(8);return;}auto re=heif_context_read_from_file(wc,filename.c_str(),nullptr);heif_image_handle*wh=nullptr;if(re.code||heif_context_get_primary_image_handle(wc,&wh).code){failed.store(9);heif_context_free(wc);return;}heif_decoding_options*opt=heif_decoding_options_alloc();opt->version=8;opt->num_codec_threads=1;for(;;){int i=next.fetch_add(1);if(i>=n||failed.load())break;unsigned tx=i%t.num_columns,ty=i/t.num_columns;heif_image*im=nullptr;auto er=heif_image_handle_decode_image_tile(wh,&im,heif_colorspace_RGB,heif_chroma_interleaved_RGB,opt,tx,ty);if(er.code||!im){failed.store(er.code?er.code:1);break;}int tw=heif_image_get_width(im,heif_channel_interleaved),th=heif_image_get_height(im,heif_channel_interleaved),stride=0;auto*src=heif_image_get_plane_readonly(im,heif_channel_interleaved,&stride);int x0=(int)tx*(int)t.tile_width+sx,y0=(int)ty*(int)t.tile_height+sy,dx0=std::max(0,x0),dy0=std::max(0,y0),dx1=std::min(w,x0+tw),dy1=std::min(hh,y0+th);for(int y=dy0;y<dy1;y++){int xx=dx0-x0;auto*sp=src+(size_t)(y-y0)*stride+(size_t)xx*3;auto*dp=dst+((size_t)y*w+dx0)*3;memcpy(dp,sp,(size_t)(dx1-dx0)*3);}heif_image_release(im);}heif_decoding_options_free(opt);heif_image_handle_release(wh);heif_context_free(wc);});
 for(auto&thd:threads)thd.join();if(failed.load()){if(message&&message_size)snprintf(message,message_size,"tile decode failed (%d)",failed.load());free(dst);heif_image_handle_release(h);heif_context_free(c);return 6;}
 *output=dst;*out_w=w;*out_h=hh;heif_image_handle_release(h);heif_context_free(c);return 0;
}
extern "C" void wave_heif_free_rgb(void*p){free(p);}
